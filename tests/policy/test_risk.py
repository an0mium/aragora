"""Tests for policy/risk.py — Risk levels, blast radius, and risk budgets."""

from __future__ import annotations

from aragora.policy.risk import (
    BLAST_RADIUS_DESCRIPTIONS,
    RISK_LEVEL_DESCRIPTIONS,
    BlastRadius,
    RiskBudget,
    RiskLevel,
    get_blast_radius_color,
    get_risk_color,
)


# =============================================================================
# Enums
# =============================================================================


class TestRiskLevel:
    def test_values(self):
        assert RiskLevel.NONE == 0
        assert RiskLevel.LOW == 1
        assert RiskLevel.MEDIUM == 2
        assert RiskLevel.HIGH == 3
        assert RiskLevel.CRITICAL == 4

    def test_ordering(self):
        assert (
            RiskLevel.NONE < RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH < RiskLevel.CRITICAL
        )

    def test_int_behavior(self):
        assert RiskLevel.MEDIUM + 1 == 3
        assert int(RiskLevel.HIGH) == 3


class TestBlastRadius:
    def test_values(self):
        assert BlastRadius.READ_ONLY == 0
        assert BlastRadius.DRAFT == 1
        assert BlastRadius.LOCAL == 2
        assert BlastRadius.SHARED == 3
        assert BlastRadius.PRODUCTION == 4

    def test_ordering(self):
        assert (
            BlastRadius.READ_ONLY
            < BlastRadius.DRAFT
            < BlastRadius.LOCAL
            < BlastRadius.SHARED
            < BlastRadius.PRODUCTION
        )


# =============================================================================
# RiskBudget — init and properties
# =============================================================================


class TestRiskBudgetInit:
    def test_defaults(self):
        b = RiskBudget()
        assert b.total == 100.0
        assert b.spent == 0.0
        assert b.human_approval_threshold == 80.0
        assert b.max_single_action == 30.0
        assert b.actions == []
        assert b.last_action_at is None
        assert b.created_at  # should be an ISO string

    def test_custom(self):
        b = RiskBudget(total=200, spent=50, human_approval_threshold=150, max_single_action=60)
        assert b.total == 200
        assert b.spent == 50
        assert b.human_approval_threshold == 150
        assert b.max_single_action == 60


class TestRiskBudgetProperties:
    def test_remaining(self):
        b = RiskBudget(total=100, spent=30)
        assert b.remaining == 70.0

    def test_remaining_cannot_go_negative(self):
        b = RiskBudget(total=100, spent=150)
        assert b.remaining == 0.0

    def test_utilization(self):
        b = RiskBudget(total=100, spent=25)
        assert b.utilization == 0.25

    def test_utilization_zero_total(self):
        b = RiskBudget(total=0, spent=0)
        assert b.utilization == 1.0

    def test_requires_human_approval_false(self):
        b = RiskBudget(total=100, spent=50, human_approval_threshold=80)
        assert b.requires_human_approval is False

    def test_requires_human_approval_true(self):
        b = RiskBudget(total=100, spent=80, human_approval_threshold=80)
        assert b.requires_human_approval is True

    def test_requires_human_approval_above(self):
        b = RiskBudget(total=100, spent=90, human_approval_threshold=80)
        assert b.requires_human_approval is True


# =============================================================================
# calculate_cost
# =============================================================================


class TestCalculateCost:
    def test_read_only_is_free(self):
        b = RiskBudget()
        cost = b.calculate_cost(RiskLevel.CRITICAL, BlastRadius.READ_ONLY)
        # risk=4, blast=0, formula = risk * (blast+1) = 4 * 1 = 4
        assert cost == 4.0

    def test_none_risk_is_free(self):
        b = RiskBudget()
        cost = b.calculate_cost(RiskLevel.NONE, BlastRadius.PRODUCTION)
        # risk=0, blast=4, formula = 0 * (4+1) = 0
        assert cost == 0.0

    def test_critical_production(self):
        b = RiskBudget()
        cost = b.calculate_cost(RiskLevel.CRITICAL, BlastRadius.PRODUCTION)
        # 4 * (4+1) = 20
        assert cost == 20.0

    def test_medium_local(self):
        b = RiskBudget()
        cost = b.calculate_cost(RiskLevel.MEDIUM, BlastRadius.LOCAL)
        # 2 * (2+1) = 6
        assert cost == 6.0

    def test_multiplier(self):
        b = RiskBudget()
        cost = b.calculate_cost(RiskLevel.LOW, BlastRadius.DRAFT, multiplier=2.5)
        # 1 * (1+1) * 2.5 = 5.0
        assert cost == 5.0

    def test_zero_multiplier(self):
        b = RiskBudget()
        cost = b.calculate_cost(RiskLevel.CRITICAL, BlastRadius.PRODUCTION, multiplier=0)
        assert cost == 0.0


# =============================================================================
# can_afford / can_afford_without_approval
# =============================================================================


class TestCanAfford:
    def test_within_budget(self):
        b = RiskBudget(total=100, spent=0, max_single_action=100)
        assert b.can_afford(50) is True

    def test_exact_budget(self):
        b = RiskBudget(total=100, spent=50, max_single_action=100)
        assert b.can_afford(50) is True

    def test_over_budget(self):
        b = RiskBudget(total=100, spent=80)
        assert b.can_afford(21) is False

    def test_exceeds_max_single_action(self):
        b = RiskBudget(total=100, spent=0, max_single_action=30)
        assert b.can_afford(31) is False

    def test_can_afford_without_approval_within(self):
        b = RiskBudget(total=100, spent=0, human_approval_threshold=80, max_single_action=100)
        assert b.can_afford_without_approval(50) is True

    def test_can_afford_without_approval_crosses_threshold(self):
        b = RiskBudget(total=100, spent=50, human_approval_threshold=80)
        assert b.can_afford_without_approval(30) is False

    def test_can_afford_without_approval_exceeds_budget(self):
        b = RiskBudget(total=100, spent=90)
        assert b.can_afford_without_approval(20) is False

    def test_can_afford_without_approval_exceeds_max_single(self):
        b = RiskBudget(total=100, spent=0, max_single_action=10)
        assert b.can_afford_without_approval(15) is False


# =============================================================================
# spend
# =============================================================================


class TestSpend:
    def test_spend_within_budget(self):
        b = RiskBudget(total=100, spent=0)
        result = b.spend(10, "write config.py")
        assert result is True
        assert b.spent == 10
        assert len(b.actions) == 1
        assert b.actions[0]["cost"] == 10
        assert b.actions[0]["description"] == "write config.py"
        assert b.actions[0]["within_budget"] is True
        assert b.last_action_at is not None

    def test_spend_over_budget(self):
        b = RiskBudget(total=100, spent=90)
        result = b.spend(20, "deploy", agent="claude", tool="shell")
        assert result is False
        assert b.spent == 110
        assert b.actions[0]["within_budget"] is False
        assert b.actions[0]["agent"] == "claude"
        assert b.actions[0]["tool"] == "shell"

    def test_spend_exceeds_max_single(self):
        b = RiskBudget(total=100, spent=0, max_single_action=5)
        result = b.spend(10, "big action")
        assert result is False

    def test_spend_tracks_remaining(self):
        b = RiskBudget(total=100, spent=0)
        b.spend(25, "action 1")
        assert b.actions[0]["remaining_after"] == 75.0

    def test_multiple_spends(self):
        b = RiskBudget(total=100, spent=0)
        b.spend(10, "a1")
        b.spend(20, "a2")
        b.spend(30, "a3")
        assert b.spent == 60
        assert len(b.actions) == 3

    def test_spend_defaults(self):
        b = RiskBudget(total=100, spent=0)
        b.spend(5, "test")
        assert b.actions[0]["agent"] == "unknown"
        assert b.actions[0]["tool"] == "unknown"


# =============================================================================
# to_dict
# =============================================================================


class TestToDict:
    def test_basic(self):
        b = RiskBudget(total=100, spent=25)
        d = b.to_dict()
        assert d["total"] == 100
        assert d["spent"] == 25
        assert d["remaining"] == 75.0
        assert d["utilization"] == 0.25
        assert d["human_approval_threshold"] == 80.0
        assert d["max_single_action"] == 30.0
        assert d["requires_human_approval"] is False
        assert d["action_count"] == 0
        assert d["created_at"] is not None
        assert d["last_action_at"] is None

    def test_after_actions(self):
        b = RiskBudget(total=100, spent=0)
        b.spend(10, "a1")
        b.spend(20, "a2")
        d = b.to_dict()
        assert d["action_count"] == 2
        assert d["spent"] == 30
        assert d["last_action_at"] is not None


# =============================================================================
# Module-level helpers
# =============================================================================


class TestDescriptions:
    def test_risk_level_descriptions(self):
        assert RiskLevel.NONE in RISK_LEVEL_DESCRIPTIONS
        assert RiskLevel.CRITICAL in RISK_LEVEL_DESCRIPTIONS
        assert len(RISK_LEVEL_DESCRIPTIONS) == 5

    def test_blast_radius_descriptions(self):
        assert BlastRadius.READ_ONLY in BLAST_RADIUS_DESCRIPTIONS
        assert BlastRadius.PRODUCTION in BLAST_RADIUS_DESCRIPTIONS
        assert len(BLAST_RADIUS_DESCRIPTIONS) == 5


class TestColorHelpers:
    def test_risk_colors(self):
        assert get_risk_color(RiskLevel.NONE) == "gray"
        assert get_risk_color(RiskLevel.LOW) == "green"
        assert get_risk_color(RiskLevel.MEDIUM) == "yellow"
        assert get_risk_color(RiskLevel.HIGH) == "orange"
        assert get_risk_color(RiskLevel.CRITICAL) == "red"

    def test_blast_radius_colors(self):
        assert get_blast_radius_color(BlastRadius.READ_ONLY) == "gray"
        assert get_blast_radius_color(BlastRadius.DRAFT) == "blue"
        assert get_blast_radius_color(BlastRadius.LOCAL) == "green"
        assert get_blast_radius_color(BlastRadius.SHARED) == "yellow"
        assert get_blast_radius_color(BlastRadius.PRODUCTION) == "red"
