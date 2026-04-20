"""Tests for aragora.review.policy — review-depth + budget contracts."""

from __future__ import annotations

import json

import pytest

from aragora.review import (
    BudgetHeadroom,
    BudgetScope,
    CostMeter,
    DepthTrigger,
    ReviewBudget,
    ReviewDepth,
    ReviewPolicy,
    ReviewPolicyDecision,
    RiskClass,
)


# --- Enums ---------------------------------------------------------------


class TestReviewDepth:
    def test_values(self) -> None:
        # Ordered cheapest -> most thorough; value strings are canonical.
        assert ReviewDepth.TRIVIAL.value == "trivial"
        assert ReviewDepth.STANDARD.value == "standard"
        assert ReviewDepth.DEEP.value == "deep"

    def test_exactly_three_levels(self) -> None:
        # Foundation posture: three levels is enough; more is YAGNI until
        # a real consumer wants a fourth.
        assert len(list(ReviewDepth)) == 3


class TestRiskClass:
    def test_values(self) -> None:
        assert RiskClass.LOW.value == "low"
        assert RiskClass.MEDIUM.value == "medium"
        assert RiskClass.HIGH.value == "high"
        assert RiskClass.CRITICAL.value == "critical"


class TestReviewPolicyDecision:
    def test_values(self) -> None:
        assert ReviewPolicyDecision.ALLOW.value == "allow"
        assert ReviewPolicyDecision.DEGRADE.value == "degrade"
        assert ReviewPolicyDecision.DENY.value == "deny"
        assert ReviewPolicyDecision.ESCALATE.value == "escalate"

    def test_degrade_exists_for_review_specific_semantics(self) -> None:
        # `DEGRADE` is the review-specific value the generic
        # aragora.policy.engine.PolicyDecision does not express. If a
        # future refactor tries to unify the enums, it must preserve
        # DEGRADE or the review substrate loses its cheapest-possible
        # fallback.
        assert ReviewPolicyDecision.DEGRADE in set(ReviewPolicyDecision)


# --- DepthTrigger --------------------------------------------------------


class TestDepthTrigger:
    def test_frozen(self) -> None:
        trigger = DepthTrigger(target_depth=ReviewDepth.DEEP)
        with pytest.raises((AttributeError, TypeError)):
            trigger.target_depth = ReviewDepth.TRIVIAL  # type: ignore[misc]

    def test_subsystem_prefixes_is_tuple(self) -> None:
        trigger = DepthTrigger(
            target_depth=ReviewDepth.DEEP,
            subsystem_prefixes=("aragora/security/", "aragora/auth/"),
        )
        assert isinstance(trigger.subsystem_prefixes, tuple)
        with pytest.raises(AttributeError):
            trigger.subsystem_prefixes.append("aragora/billing/")  # type: ignore[attr-defined]

    def test_to_dict_serializes_enums_and_tuples(self) -> None:
        trigger = DepthTrigger(
            target_depth=ReviewDepth.DEEP,
            min_additions_plus_deletions=500,
            subsystem_prefixes=("aragora/security/",),
            min_risk_class=RiskClass.HIGH,
        )
        d = trigger.to_dict()
        assert d["target_depth"] == "deep"
        assert d["subsystem_prefixes"] == ["aragora/security/"]
        assert d["min_risk_class"] == "high"
        assert d["min_additions_plus_deletions"] == 500

    def test_min_risk_class_omitted_when_none(self) -> None:
        trigger = DepthTrigger(target_depth=ReviewDepth.TRIVIAL)
        d = trigger.to_dict()
        # asdict preserves None in the dict; to_dict only overrides when set.
        assert d.get("min_risk_class") is None


# --- ReviewBudget --------------------------------------------------------


class TestReviewBudget:
    def test_defaults_are_dogfood_safe(self) -> None:
        # Per #6305 acceptance: "Safe defaults for Aragora dogfood before
        # wider rollout." Tests lock these so wider rollout requires an
        # explicit rewrite, not silent drift.
        budget = ReviewBudget()
        assert budget.per_pr_usd_cap == 25.0  # Anthropic market anchor
        assert budget.alert_threshold_pct == 80.0
        assert budget.hard_limit is True
        assert budget.per_repo_usd_daily_cap == 0.0  # 0 = unlimited
        assert budget.per_org_usd_daily_cap == 0.0

    def test_daily_caps_are_depth_scoped_by_default(self) -> None:
        # Per #6305 AC: "Per-repo / per-org caps for deep review runs."
        # Default scoping: daily caps apply at STANDARD+; TRIVIAL
        # reviews don't count against the daily pool. Prevents a flood
        # of typo-only reviews from exhausting the daily budget and
        # then denying a legitimate DEEP review later in the day.
        budget = ReviewBudget()
        assert budget.daily_caps_apply_at_or_above_depth == ReviewDepth.STANDARD

    def test_daily_cap_scoping_is_configurable(self) -> None:
        # Teams that want the daily pools to cover STANDARD reviews too
        # can set this to STANDARD; teams that only want DEEP runs to
        # count against the daily pool can set it to DEEP.
        budget = ReviewBudget(daily_caps_apply_at_or_above_depth=ReviewDepth.DEEP)
        assert budget.daily_caps_apply_at_or_above_depth == ReviewDepth.DEEP

    def test_per_pr_cap_is_not_depth_scoped(self) -> None:
        # Per-PR cap applies to EVERY run regardless of depth. Only the
        # daily repo/org pools carry the depth scoping. If a future
        # refactor tries to add `per_pr_applies_at_or_above_depth`, this
        # test fails loudly.
        budget = ReviewBudget()
        assert not hasattr(budget, "per_pr_applies_at_or_above_depth")

    def test_to_dict_serializes_depth_enum(self) -> None:
        budget = ReviewBudget(daily_caps_apply_at_or_above_depth=ReviewDepth.DEEP)
        d = budget.to_dict()
        assert d["daily_caps_apply_at_or_above_depth"] == "deep"

    def test_frozen(self) -> None:
        budget = ReviewBudget()
        with pytest.raises((AttributeError, TypeError)):
            budget.per_pr_usd_cap = 100.0  # type: ignore[misc]

    def test_to_dict_roundtrip(self) -> None:
        budget = ReviewBudget(
            per_pr_usd_cap=50.0,
            per_repo_usd_daily_cap=500.0,
            alert_threshold_pct=75.0,
        )
        roundtrip = json.loads(json.dumps(budget.to_dict()))
        assert roundtrip["per_pr_usd_cap"] == 50.0
        assert roundtrip["per_repo_usd_daily_cap"] == 500.0
        assert roundtrip["alert_threshold_pct"] == 75.0
        assert roundtrip["daily_caps_apply_at_or_above_depth"] == "standard"


# --- ReviewPolicy --------------------------------------------------------


class TestReviewPolicy:
    def test_defaults(self) -> None:
        policy = ReviewPolicy()
        assert policy.default_depth == ReviewDepth.STANDARD
        assert policy.depth_rules == ()
        # Nested dataclass default uses dogfood-safe budget.
        assert policy.budget.per_pr_usd_cap == 25.0

    def test_depth_rules_is_immutable_tuple(self) -> None:
        policy = ReviewPolicy(
            depth_rules=(
                DepthTrigger(target_depth=ReviewDepth.DEEP, min_additions_plus_deletions=500),
            ),
        )
        assert isinstance(policy.depth_rules, tuple)
        with pytest.raises(AttributeError):
            policy.depth_rules.append(  # type: ignore[attr-defined]
                DepthTrigger(target_depth=ReviewDepth.TRIVIAL)
            )

    def test_frozen(self) -> None:
        policy = ReviewPolicy()
        with pytest.raises((AttributeError, TypeError)):
            policy.default_depth = ReviewDepth.DEEP  # type: ignore[misc]

    def test_to_dict_nests_budget_and_rules(self) -> None:
        policy = ReviewPolicy(
            depth_rules=(
                DepthTrigger(
                    target_depth=ReviewDepth.DEEP,
                    subsystem_prefixes=("aragora/security/",),
                    min_risk_class=RiskClass.HIGH,
                ),
            ),
            default_depth=ReviewDepth.STANDARD,
        )
        d = policy.to_dict()
        assert d["default_depth"] == "standard"
        assert d["budget"]["per_pr_usd_cap"] == 25.0
        assert d["depth_rules"][0]["target_depth"] == "deep"
        assert d["depth_rules"][0]["subsystem_prefixes"] == ["aragora/security/"]

    def test_json_roundtrip(self) -> None:
        policy = ReviewPolicy()
        roundtrip = json.loads(json.dumps(policy.to_dict()))
        assert roundtrip["default_depth"] == "standard"
        assert roundtrip["budget"]["per_pr_usd_cap"] == 25.0


# --- BudgetScope + BudgetHeadroom -----------------------------------------


class TestBudgetScope:
    def test_values(self) -> None:
        assert BudgetScope.PER_PR.value == "per_pr"
        assert BudgetScope.PER_REPO_DAILY.value == "per_repo_daily"
        assert BudgetScope.PER_ORG_DAILY.value == "per_org_daily"

    def test_three_scopes_match_review_budget_pools(self) -> None:
        # BudgetScope must cover exactly the pools ReviewBudget defines.
        # If a future ReviewBudget adds a fourth pool without extending
        # BudgetScope, CostMeter would lose the ability to identify it.
        budget = ReviewBudget()
        pool_fields = {"per_pr_usd_cap", "per_repo_usd_daily_cap", "per_org_usd_daily_cap"}
        for field_name in pool_fields:
            assert hasattr(budget, field_name)
        assert len(list(BudgetScope)) == 3


class TestBudgetHeadroom:
    def test_frozen(self) -> None:
        h = BudgetHeadroom(scope=BudgetScope.PER_PR, cap_usd=25.0, remaining_usd=20.0)
        with pytest.raises((AttributeError, TypeError)):
            h.remaining_usd = 0.0  # type: ignore[misc]

    def test_to_dict_serializes_scope(self) -> None:
        h = BudgetHeadroom(
            scope=BudgetScope.PER_REPO_DAILY,
            cap_usd=500.0,
            remaining_usd=350.0,
            applies_at_or_above_depth=ReviewDepth.STANDARD,
        )
        d = h.to_dict()
        assert d["scope"] == "per_repo_daily"
        assert d["applies_at_or_above_depth"] == "standard"
        assert d["cap_usd"] == 500.0
        assert d["remaining_usd"] == 350.0

    def test_per_pr_scope_has_no_depth_scoping(self) -> None:
        # Per-PR cap applies to all depths; applies_at_or_above_depth=None
        # is the canonical representation.
        h = BudgetHeadroom(scope=BudgetScope.PER_PR, cap_usd=25.0, remaining_usd=15.0)
        assert h.applies_at_or_above_depth is None


# --- CostMeter ------------------------------------------------------------


class TestCostMeter:
    def _meter(self, **overrides) -> CostMeter:
        defaults = dict(
            depth_chosen=ReviewDepth.STANDARD,
            decision=ReviewPolicyDecision.ALLOW,
            estimated_cost_usd=0.25,
            actual_cost_usd=0.24,
            headroom_by_scope=(
                BudgetHeadroom(scope=BudgetScope.PER_PR, cap_usd=25.0, remaining_usd=24.76),
            ),
        )
        defaults.update(overrides)
        return CostMeter(**defaults)

    def test_frozen(self) -> None:
        meter = self._meter()
        with pytest.raises((AttributeError, TypeError)):
            meter.depth_chosen = ReviewDepth.DEEP  # type: ignore[misc]

    def test_to_dict_serializes_enums(self) -> None:
        meter = self._meter(
            depth_chosen=ReviewDepth.DEEP,
            decision=ReviewPolicyDecision.DEGRADE,
            estimated_cost_usd=8.0,
            actual_cost_usd=7.5,
            alert_triggered=True,
        )
        d = meter.to_dict()
        assert d["depth_chosen"] == "deep"
        assert d["decision"] == "degrade"
        assert d["alert_triggered"] is True

    def test_addresses_packet_cost_meter_acceptance_criterion(self) -> None:
        # Per #6305 acceptance: "Packet output includes cost used and
        # budget context." CostMeter is the exact shape the future
        # packet renderer will embed.
        meter = self._meter()
        d = meter.to_dict()
        # "cost used" — actual_cost_usd
        assert "actual_cost_usd" in d
        # "budget context" — per-pool headroom (not just one number)
        assert "headroom_by_scope" in d
        assert isinstance(d["headroom_by_scope"], list)

    def test_binding_scope_identifies_which_pool_was_limiting(self) -> None:
        # Per codex's P1 finding on #6359: packet-readers must be able
        # to tell which pool forced a DEGRADE/DENY decision. binding_scope
        # is that dimension.
        meter = self._meter(
            depth_chosen=ReviewDepth.STANDARD,  # runner wanted DEEP but degraded
            decision=ReviewPolicyDecision.DEGRADE,
            headroom_by_scope=(
                BudgetHeadroom(scope=BudgetScope.PER_PR, cap_usd=25.0, remaining_usd=24.0),
                BudgetHeadroom(
                    scope=BudgetScope.PER_REPO_DAILY,
                    cap_usd=50.0,
                    remaining_usd=2.0,  # nearly exhausted
                    applies_at_or_above_depth=ReviewDepth.STANDARD,
                ),
            ),
            binding_scope=BudgetScope.PER_REPO_DAILY,
        )
        d = meter.to_dict()
        assert d["binding_scope"] == "per_repo_daily"
        # Per-pool fields preserved so UI can show "repo pool: $2 left of $50"
        assert len(d["headroom_by_scope"]) == 2
        assert d["headroom_by_scope"][1]["scope"] == "per_repo_daily"
        assert d["headroom_by_scope"][1]["remaining_usd"] == 2.0

    def test_binding_scope_optional_for_allow_decisions(self) -> None:
        # When a run is ALLOWED with no pool near its cap, binding_scope
        # is None — there's no "limit that made the call."
        meter = self._meter()
        assert meter.binding_scope is None
        d = meter.to_dict()
        assert d.get("binding_scope") is None

    def test_headroom_by_scope_is_immutable_tuple(self) -> None:
        meter = self._meter()
        assert isinstance(meter.headroom_by_scope, tuple)
        with pytest.raises(AttributeError):
            meter.headroom_by_scope.append(  # type: ignore[attr-defined]
                BudgetHeadroom(scope=BudgetScope.PER_ORG_DAILY, cap_usd=1000.0, remaining_usd=500.0)
            )

    def test_json_roundtrip(self) -> None:
        meter = self._meter()
        roundtrip = json.loads(json.dumps(meter.to_dict()))
        assert roundtrip["depth_chosen"] == "standard"
        assert roundtrip["decision"] == "allow"
        assert roundtrip["headroom_by_scope"][0]["scope"] == "per_pr"


# --- Cross-module composition -------------------------------------------


class TestContractComposition:
    def test_review_policy_decision_disjoint_from_engine_policy_decision(self) -> None:
        # Intentional non-reuse: aragora.policy.engine.PolicyDecision has
        # ALLOW/DENY/ESCALATE/BUDGET_EXCEEDED for deployment decisions.
        # ReviewPolicyDecision has ALLOW/DEGRADE/DENY/ESCALATE for review
        # runs. They share three strings but diverge on the fourth, which
        # is why review has its own enum. This test documents that the
        # divergence is intentional.
        from aragora.policy.engine import PolicyDecision as GenericPolicyDecision

        review_values = {d.value for d in ReviewPolicyDecision}
        generic_values = {d.value for d in GenericPolicyDecision}
        # Review has DEGRADE, generic does not.
        assert "degrade" in review_values
        assert "degrade" not in generic_values
        # Generic has BUDGET_EXCEEDED, review does not (review uses DENY
        # with a reason for the same signal).
        assert "budget_exceeded" in generic_values
        assert "budget_exceeded" not in review_values

    def test_review_budget_does_not_replace_generic_budget_policy(self) -> None:
        # Intentional non-replacement: aragora.billing.budget_policy.BudgetPolicy
        # is the generic workspace budget (monthly/daily/per-debate).
        # ReviewBudget is the PR-review slice (per-pr/per-repo/per-org).
        # They compose; neither replaces the other.
        from aragora.billing.budget_policy import BudgetPolicy

        budget = ReviewBudget()
        generic = BudgetPolicy()
        # ReviewBudget has per_pr_usd_cap; generic does not.
        assert hasattr(budget, "per_pr_usd_cap")
        assert not hasattr(generic, "per_pr_usd_cap")
        # Generic has monthly_limit; ReviewBudget does not (that's a
        # workspace-level concern, not per-review).
        assert hasattr(generic, "monthly_limit")
        assert not hasattr(budget, "monthly_limit")
