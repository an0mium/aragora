"""Tests for the pure settlement routing/gating decision (settle_pr.py's brain).

`plan_settlement` decides, from already-fetched quorum/tier state, whether a PR
can be driven to settlement and by which route (auto-merge for Tier 0-2, operator
human-settlement for Tier 3-4) -- accumulating every blocker. It performs no I/O,
so it is fully unit-testable here. `summarize_collect` flattens a collect-evidence
JSON payload into the fields the planner + diagnostics need.
"""

from __future__ import annotations

from aragora.swarm.settle_plan import (
    ROUTE_AUTO_MERGE,
    ROUTE_BLOCKED,
    ROUTE_OPERATOR_TIER4,
    plan_settlement,
    summarize_collect,
    tier4_settle_commands,
)


def test_tier_two_satisfied_routes_to_auto_merge():
    plan = plan_settlement(tier=2, quorum_satisfied=True, supportive_families=["claude", "grok"])
    assert plan.route == ROUTE_AUTO_MERGE
    assert plan.ready_to_mutate is True
    assert plan.requires_operator_login is False
    assert plan.blockers == ()


def test_quorum_unsatisfied_blocks_and_is_not_ready():
    plan = plan_settlement(tier=2, quorum_satisfied=False, supportive_families=["grok"])
    assert plan.ready_to_mutate is False
    assert any("quorum not satisfied" in b for b in plan.blockers)


def test_tier_two_unresolved_dissent_blocks_auto_merge():
    plan = plan_settlement(
        tier=1,
        quorum_satisfied=True,
        supportive_families=["claude", "grok"],
        unresolved_dissent=True,
    )
    assert plan.route == ROUTE_AUTO_MERGE
    assert plan.ready_to_mutate is False
    assert any("dissent" in b for b in plan.blockers)


def test_tier_four_requires_operator_login():
    # Without --operator-login the Tier-4 human settlement cannot proceed.
    plan = plan_settlement(tier=4, quorum_satisfied=True, supportive_families=["claude", "grok"])
    assert plan.route == ROUTE_OPERATOR_TIER4
    assert plan.requires_operator_login is True
    assert plan.ready_to_mutate is False
    assert any("--operator-login" in b for b in plan.blockers)


def test_tier_four_unresolved_dissent_blocks_even_with_login():
    # settle_tier4_pr hard-fails on unresolved dissent, so a Tier 3-4 plan must
    # also block on it (not just Tier 0-2) -- else it surfaces doomed commands.
    plan = plan_settlement(
        tier=4,
        quorum_satisfied=True,
        supportive_families=["claude", "grok"],
        unresolved_dissent=True,
        operator_login_provided=True,
    )
    assert plan.route == ROUTE_OPERATOR_TIER4
    assert plan.ready_to_mutate is False
    assert any("dissent" in b for b in plan.blockers)


def test_tier_four_with_operator_login_is_ready():
    plan = plan_settlement(
        tier=3,
        quorum_satisfied=True,
        supportive_families=["claude", "grok"],
        operator_login_provided=True,
    )
    assert plan.route == ROUTE_OPERATOR_TIER4
    assert plan.ready_to_mutate is True
    assert plan.blockers == ()


def test_unknown_tier_is_blocked_fail_safe():
    plan = plan_settlement(tier=None, quorum_satisfied=True, supportive_families=["claude", "grok"])
    assert plan.route == ROUTE_BLOCKED
    assert plan.ready_to_mutate is False
    assert any("tier unknown" in b for b in plan.blockers)


def test_summarize_collect_flattens_payload():
    payload = {
        "tier": 2,
        "head_sha": "abc123",
        "has_supportive_quorum": True,
        "supportive_families": ["claude", "grok"],
        "dissenting_families": [],
        "items": [
            {
                "family": "claude",
                "verdict": "pass",
                "would_count": True,
                "counted_reviewer_ids": ["claude"],
                "problems": [],
            },
            {
                "family": "grok",
                "verdict": "pass",
                "would_count": True,
                "counted_reviewer_ids": ["grok"],
                "problems": [],
            },
        ],
        "failures": [],
    }
    s = summarize_collect(payload)
    assert s["tier"] == 2
    assert s["quorum_satisfied"] is True
    assert s["supportive_families"] == ["claude", "grok"]
    assert s["dissenting_families"] == []
    assert len(s["items"]) == 2
    assert s["failures"] == []


def test_summarize_collect_handles_error_envelope():
    s = summarize_collect({"mode": "collect_evidence", "error": "boom"})
    assert s["error"] == "boom"
    assert s["quorum_satisfied"] is False
    assert s["tier"] is None


def test_tier4_settle_commands_are_head_bound_and_ordered():
    cmds = tier4_settle_commands(repo="owner/repo", pr=42, head="abc123", operator_login="alice")
    assert len(cmds) == 3
    # check -> settle-only -> merge-apply, every command head-bound + operator-pinned.
    assert cmds[0].endswith("--check")
    assert cmds[1].endswith("--settle-only")
    assert cmds[2].endswith("--merge-apply")
    for c in cmds:
        assert "--pr 42" in c
        assert "--head abc123" in c
        assert "--trusted-operator-login alice" in c
        assert "--repo owner/repo" in c
    # no app-token prefix unless requested
    assert not cmds[2].startswith("ARAGORA_DISABLE_GITHUB_APP_TOKEN=1")


def test_tier4_settle_commands_no_app_token_prefixes_merge_only():
    cmds = tier4_settle_commands(
        repo="owner/repo", pr=42, head="abc123", operator_login="alice", no_app_token=True
    )
    assert cmds[2].startswith("ARAGORA_DISABLE_GITHUB_APP_TOKEN=1 ")
    # only the irreversible merge-apply step carries the override
    assert not cmds[0].startswith("ARAGORA_DISABLE_GITHUB_APP_TOKEN=1")
    assert not cmds[1].startswith("ARAGORA_DISABLE_GITHUB_APP_TOKEN=1")
