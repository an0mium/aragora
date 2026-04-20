"""Tests for aragora.review.policy — review-depth + budget contracts."""

from __future__ import annotations

import json

import pytest

from aragora.review import (
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


# --- CostMeter ------------------------------------------------------------


class TestCostMeter:
    def test_frozen(self) -> None:
        meter = CostMeter(
            depth_chosen=ReviewDepth.STANDARD,
            decision=ReviewPolicyDecision.ALLOW,
            estimated_cost_usd=0.25,
            actual_cost_usd=0.24,
            budget_remaining_usd=24.76,
            per_pr_cap_usd=25.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            meter.depth_chosen = ReviewDepth.DEEP  # type: ignore[misc]

    def test_to_dict_serializes_enums_as_strings(self) -> None:
        meter = CostMeter(
            depth_chosen=ReviewDepth.DEEP,
            decision=ReviewPolicyDecision.DEGRADE,
            estimated_cost_usd=8.0,
            actual_cost_usd=7.5,
            budget_remaining_usd=17.5,
            per_pr_cap_usd=25.0,
            alert_triggered=True,
        )
        d = meter.to_dict()
        assert d["depth_chosen"] == "deep"
        assert d["decision"] == "degrade"
        assert d["alert_triggered"] is True

    def test_addresses_packet_cost_meter_acceptance_criterion(self) -> None:
        # Per #6305 acceptance: "Packet output includes cost used and
        # budget context." CostMeter is the exact shape the future packet
        # renderer will embed.
        meter = CostMeter(
            depth_chosen=ReviewDepth.STANDARD,
            decision=ReviewPolicyDecision.ALLOW,
            estimated_cost_usd=0.25,
            actual_cost_usd=0.24,
            budget_remaining_usd=24.76,
            per_pr_cap_usd=25.0,
        )
        d = meter.to_dict()
        # "cost used" — actual_cost_usd
        assert "actual_cost_usd" in d
        # "budget context" — budget_remaining_usd + per_pr_cap_usd
        assert "budget_remaining_usd" in d
        assert "per_pr_cap_usd" in d

    def test_json_roundtrip(self) -> None:
        meter = CostMeter(
            depth_chosen=ReviewDepth.STANDARD,
            decision=ReviewPolicyDecision.ALLOW,
            estimated_cost_usd=0.5,
            actual_cost_usd=0.45,
            budget_remaining_usd=24.55,
            per_pr_cap_usd=25.0,
        )
        roundtrip = json.loads(json.dumps(meter.to_dict()))
        assert roundtrip["depth_chosen"] == "standard"
        assert roundtrip["decision"] == "allow"
        assert roundtrip["per_pr_cap_usd"] == 25.0


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
