"""Tests for ``aragora.policy.delegation_contract`` — Delegation Contract v0.1.

Covers per the spec:
- root-contract validation (self-consistency)
- monotonic narrowing rules for parent→child issuance
- scope-widening rejection
- budget-debit math
- max-depth rejection
- destructive-action human gate (v0.1 effectively denies destructive)
- expires_at narrowing
- surface containment
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aragora.policy import (
    CONTRACT_SCHEMA_VERSION,
    GOAL_SPEC_SCHEMA_VERSION,
    AcceptanceCriterion,
    AllowedSurfaces,
    ContractBudget,
    ContractValidationError,
    DelegationContract,
    GoalSpec,
    RiskBudget,
    make_root_contract,
    narrow_for_child,
)


# ---------- helpers ----------


def _now_iso(offset_minutes: int = 0) -> str:
    return (datetime.now(UTC) + timedelta(minutes=offset_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_simple_parent(**overrides) -> DelegationContract:
    defaults = dict(
        contract_id="root-1",
        root_intent_id="intent-1",
        delegator="an0mium",
        delegatee="claude-A",
        goal_id="G-foo",
        allowed_actions=("read:*", "write:branch:claude/*", "spawn:subagent"),
        max_depth=3,
        duration_minutes=120,
        stale_threshold_minutes=30,
        destructive_action_policy="deny",
    )
    defaults.update(overrides)
    return make_root_contract(**defaults)


# ---------- root validation ----------


def test_root_contract_validates() -> None:
    c = _make_simple_parent()
    c.validate()
    assert c.schema_version == CONTRACT_SCHEMA_VERSION
    assert c.parent_contract_id is None
    assert c.max_depth == 3


def test_root_contract_requires_goal_id() -> None:
    with pytest.raises(ContractValidationError, match="goal_id"):
        _make_simple_parent(goal_id="")


def test_root_contract_requires_future_expiry() -> None:
    with pytest.raises(ContractValidationError, match="expires_at"):
        _make_simple_parent(duration_minutes=-1)


def test_root_contract_rejects_destructive_policy_allow() -> None:
    with pytest.raises(ContractValidationError, match="'allow' is not permitted"):
        _make_simple_parent(destructive_action_policy="allow")


def test_root_contract_rejects_v01_signature() -> None:
    parent = _make_simple_parent()
    forged = DelegationContract(
        **{**parent.__dict__, "signature": "deadbeef"},
    )
    with pytest.raises(ContractValidationError, match="signature must be None in v0.1"):
        forged.validate()


# ---------- monotonic narrowing — actions ----------


def test_child_actions_subset_ok() -> None:
    parent = _make_simple_parent(
        allowed_actions=("read:*", "write:branch:claude/*", "spawn:subagent")
    )
    child = narrow_for_child(
        parent,
        child_contract_id="child-1",
        child_delegatee="claude-B-worker",
        child_allowed_actions=("read:*",),
    )
    assert "read:*" in child.allowed_actions
    assert "write:branch:claude/*" not in child.allowed_actions


def test_child_actions_widening_rejected() -> None:
    parent = _make_simple_parent(allowed_actions=("read:*",))
    with pytest.raises(ContractValidationError, match="widen parent"):
        narrow_for_child(
            parent,
            child_contract_id="child-1",
            child_delegatee="claude-B",
            child_allowed_actions=("read:*", "write:branch:claude/*"),
        )


def test_child_denied_actions_narrowing_rejected() -> None:
    parent = _make_simple_parent(
        allowed_actions=("read:*",),
    )
    parent = DelegationContract(**{**parent.__dict__, "denied_actions": frozenset({"merge:*"})})
    with pytest.raises(ContractValidationError, match="missing"):
        narrow_for_child(
            parent,
            child_contract_id="child-1",
            child_delegatee="claude-B",
            child_denied_actions=frozenset(),  # tries to remove parent's deny
        )


# ---------- monotonic narrowing — depth ----------


def test_child_depth_decrements_by_one() -> None:
    parent = _make_simple_parent(max_depth=2)
    child = narrow_for_child(parent, child_contract_id="c1", child_delegatee="claude-B")
    assert child.max_depth == 1


def test_leaf_parent_cannot_spawn() -> None:
    parent = _make_simple_parent(max_depth=0)
    with pytest.raises(ContractValidationError, match="cannot spawn children"):
        narrow_for_child(parent, child_contract_id="c1", child_delegatee="claude-B")


# ---------- monotonic narrowing — budget ----------


def test_child_budget_within_parent_ok() -> None:
    parent_budget = ContractBudget(
        risk_budget=RiskBudget(total=100),
        max_wall_clock_minutes=120,
        max_subagents_spawned=4,
        max_prs_opened=6,
        max_api_dollars=5.00,
    )
    parent = _make_simple_parent(budget=parent_budget)
    child = narrow_for_child(
        parent,
        child_contract_id="c1",
        child_delegatee="claude-B",
        child_budget=ContractBudget(
            risk_budget=RiskBudget(total=20),
            max_wall_clock_minutes=30,
            max_subagents_spawned=0,
            max_prs_opened=1,
            max_api_dollars=0.50,
        ),
    )
    assert child.budget.max_wall_clock_minutes == 30
    assert child.budget.risk_budget.total == 20


def test_child_budget_exceeding_parent_rejected() -> None:
    parent = _make_simple_parent()
    with pytest.raises(ContractValidationError, match="exceeds parent"):
        narrow_for_child(
            parent,
            child_contract_id="c1",
            child_delegatee="claude-B",
            child_budget=ContractBudget(
                risk_budget=RiskBudget(total=10),
                max_wall_clock_minutes=parent.budget.max_wall_clock_minutes + 1,
                max_subagents_spawned=0,
                max_prs_opened=1,
                max_api_dollars=0.50,
            ),
        )


def test_child_api_dollars_exceeds_parent_rejected() -> None:
    parent = _make_simple_parent(
        budget=ContractBudget(risk_budget=RiskBudget(), max_api_dollars=1.0)
    )
    with pytest.raises(ContractValidationError, match="max_api_dollars"):
        narrow_for_child(
            parent,
            child_contract_id="c1",
            child_delegatee="claude-B",
            child_budget=ContractBudget(risk_budget=RiskBudget(total=10), max_api_dollars=2.0),
        )


def test_child_risk_total_exceeds_parent_remaining_rejected() -> None:
    parent_rb = RiskBudget(total=10, spent=5)  # remaining = 5
    parent = _make_simple_parent(budget=ContractBudget(risk_budget=parent_rb))
    with pytest.raises(ContractValidationError, match="risk_budget.total"):
        narrow_for_child(
            parent,
            child_contract_id="c1",
            child_delegatee="claude-B",
            child_budget=ContractBudget(risk_budget=RiskBudget(total=6)),
        )


# ---------- monotonic narrowing — destructive policy ----------


def test_child_destructive_policy_widening_rejected_deny_to_human_only() -> None:
    parent = _make_simple_parent(destructive_action_policy="deny")
    with pytest.raises(ContractValidationError, match="destructive_action_policy"):
        narrow_for_child(
            parent,
            child_contract_id="c1",
            child_delegatee="claude-B",
            child_destructive_policy="human-only",
        )


def test_child_destructive_policy_unchanged_ok() -> None:
    parent = _make_simple_parent(destructive_action_policy="deny")
    child = narrow_for_child(
        parent,
        child_contract_id="c1",
        child_delegatee="claude-B",
        child_destructive_policy="deny",
    )
    assert child.destructive_action_policy == "deny"


# ---------- monotonic narrowing — surfaces ----------


def test_child_branch_glob_inside_parent_ok() -> None:
    parent = _make_simple_parent(
        allowed_surfaces=AllowedSurfaces(branch_globs=frozenset({"claude/*"}))
    )
    child = narrow_for_child(
        parent,
        child_contract_id="c1",
        child_delegatee="claude-B",
        child_allowed_surfaces=AllowedSurfaces(branch_globs=frozenset({"claude/R*"})),
    )
    assert "claude/R*" in child.allowed_surfaces.branch_globs


def test_child_branch_glob_outside_parent_rejected() -> None:
    parent = _make_simple_parent(
        allowed_surfaces=AllowedSurfaces(branch_globs=frozenset({"claude/*"}))
    )
    with pytest.raises(ContractValidationError, match="branch_globs"):
        narrow_for_child(
            parent,
            child_contract_id="c1",
            child_delegatee="claude-B",
            child_allowed_surfaces=AllowedSurfaces(branch_globs=frozenset({"codex/*"})),
        )


def test_child_deny_file_globs_must_include_parent_denies() -> None:
    parent = _make_simple_parent(
        allowed_surfaces=AllowedSurfaces(
            deny_file_globs=frozenset({"CLAUDE.md", "aragora/__init__.py"})
        )
    )
    with pytest.raises(ContractValidationError, match="deny_file_globs"):
        narrow_for_child(
            parent,
            child_contract_id="c1",
            child_delegatee="claude-B",
            child_allowed_surfaces=AllowedSurfaces(
                deny_file_globs=frozenset({"CLAUDE.md"})  # missing aragora/__init__.py
            ),
        )


# ---------- monotonic narrowing — time ----------


def test_child_expires_after_parent_rejected() -> None:
    parent = _make_simple_parent(duration_minutes=30)
    later = _now_iso(offset_minutes=60)
    with pytest.raises(ContractValidationError, match="expires_at"):
        narrow_for_child(
            parent,
            child_contract_id="c1",
            child_delegatee="claude-B",
            child_expires_at=later,
        )


def test_child_stale_threshold_widening_rejected() -> None:
    parent = _make_simple_parent(stale_threshold_minutes=30)
    with pytest.raises(ContractValidationError, match="stale_threshold_minutes"):
        narrow_for_child(
            parent,
            child_contract_id="c1",
            child_delegatee="claude-B",
            child_stale_threshold_minutes=60,
        )


# ---------- chain semantics ----------


def test_child_inherits_root_intent_and_goal() -> None:
    parent = _make_simple_parent()
    child = narrow_for_child(parent, child_contract_id="c1", child_delegatee="claude-B")
    assert child.root_intent_id == parent.root_intent_id
    assert child.goal_id == parent.goal_id
    assert child.parent_contract_id == parent.contract_id


def test_grandchild_depth_chain() -> None:
    parent = _make_simple_parent(max_depth=3)
    child = narrow_for_child(parent, child_contract_id="c1", child_delegatee="claude-B")
    grandchild = narrow_for_child(child, child_contract_id="c2", child_delegatee="claude-C")
    assert grandchild.max_depth == 1
    assert grandchild.parent_contract_id == child.contract_id
    # Eventually max_depth=0 means leaf
    leaf = narrow_for_child(grandchild, child_contract_id="c3", child_delegatee="claude-D")
    assert leaf.max_depth == 0
    with pytest.raises(ContractValidationError, match="cannot spawn"):
        narrow_for_child(leaf, child_contract_id="c4", child_delegatee="claude-E")


# ---------- action filter (permits_action) ----------


def test_permits_action_matches_allowed() -> None:
    c = _make_simple_parent(allowed_actions=("read:*", "write:branch:claude/*"))
    allowed, _ = c.permits_action("read:file")
    assert allowed
    allowed, _ = c.permits_action("write:branch:claude/R02-xxx")
    assert allowed


def test_permits_action_rejects_outside_allowed() -> None:
    c = _make_simple_parent(allowed_actions=("read:*",))
    allowed, reason = c.permits_action("write:branch:claude/foo")
    assert not allowed
    assert "no allowed pattern" in reason


def test_permits_action_denied_takes_precedence() -> None:
    parent = _make_simple_parent(allowed_actions=("read:*", "write:branch:claude/*"))
    contract = DelegationContract(
        **{**parent.__dict__, "denied_actions": frozenset({"write:branch:claude/main"})}
    )
    allowed, reason = contract.permits_action("write:branch:claude/main")
    assert not allowed
    assert "denied pattern" in reason


def test_permits_action_destructive_with_deny_policy_blocked() -> None:
    """A clean contract under deny policy must block destructive actions at
    permits_action time too (defense in depth: validator catches them at
    construction, permits_action catches them at evaluation)."""
    c = _make_simple_parent(
        allowed_actions=("read:*",),
        destructive_action_policy="deny",
    )
    allowed, reason = c.permits_action("force-push:branch:foo")
    assert not allowed
    assert "destructive" in reason


def test_root_contract_rejects_destructive_in_allowed_under_deny_policy() -> None:
    with pytest.raises(ContractValidationError, match="destructive"):
        _make_simple_parent(
            allowed_actions=("force-push:branch:claude/foo",),
            destructive_action_policy="deny",
        )


def test_empty_allowed_actions_denies_by_default() -> None:
    c = _make_simple_parent(allowed_actions=())
    allowed, reason = c.permits_action("read:file")
    assert not allowed
    assert "deny by default" in reason


# ---------- expiry ----------


def test_is_expired_future() -> None:
    c = _make_simple_parent(duration_minutes=10)
    assert not c.is_expired()


def test_is_expired_with_past_time() -> None:
    c = _make_simple_parent(duration_minutes=10)
    future = datetime.now(UTC) + timedelta(minutes=20)
    assert c.is_expired(now=future)
