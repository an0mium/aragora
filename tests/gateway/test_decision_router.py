"""
Comprehensive tests for the DecisionRouter (aragora.gateway.decision_router).

Tests cover:
- Enums: RouteDestination, RiskLevel, ActionCategory, RoutingEventType
- Data classes: RoutingCriteria, RouteDecision, RoutingRule, TenantRoutingConfig,
  CategoryDefaults, RoutingMetrics, RoutingAuditEntry
- DecisionRouter: routing logic, rule management, tenant configuration,
  metrics, audit logging, anomaly detection, event handlers
- SimpleAnomalyDetector
- DEFAULT_CATEGORY_CONFIGS
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from aragora.gateway.decision_router import (
    DEFAULT_CATEGORY_CONFIGS,
    ActionCategory,
    CategoryDefaults,
    DecisionRouter,
    RiskLevel,
    RouteDecision,
    RouteDestination,
    RoutingAuditEntry,
    RoutingCriteria,
    RoutingEventType,
    RoutingMetrics,
    RoutingRule,
    SimpleAnomalyDetector,
    TenantRoutingConfig,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestRouteDestination:
    """Tests for RouteDestination enum."""

    def test_debate_value(self):
        assert RouteDestination.DEBATE.value == "debate"

    def test_execute_value(self):
        assert RouteDestination.EXECUTE.value == "execute"

    def test_hybrid_debate_then_execute_value(self):
        assert RouteDestination.HYBRID_DEBATE_THEN_EXECUTE.value == "hybrid_debate_execute"

    def test_hybrid_execute_with_validation_value(self):
        assert RouteDestination.HYBRID_EXECUTE_WITH_VALIDATION.value == "hybrid_execute_validate"

    def test_reject_value(self):
        assert RouteDestination.REJECT.value == "reject"

    def test_is_string_enum(self):
        assert isinstance(RouteDestination.DEBATE, str)
        assert RouteDestination.DEBATE == "debate"

    def test_construct_from_value(self):
        assert RouteDestination("debate") == RouteDestination.DEBATE


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_values(self):
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_is_string_enum(self):
        assert isinstance(RiskLevel.HIGH, str)
        assert RiskLevel.HIGH == "high"


class TestActionCategory:
    """Tests for ActionCategory enum."""

    def test_all_categories_present(self):
        expected = {
            "financial",
            "compliance",
            "security",
            "infrastructure",
            "data_management",
            "user_management",
            "communication",
            "analytics",
            "general",
        }
        actual = {cat.value for cat in ActionCategory}
        assert actual == expected

    def test_is_string_enum(self):
        assert isinstance(ActionCategory.FINANCIAL, str)


class TestRoutingEventType:
    """Tests for RoutingEventType enum."""

    def test_all_event_types_present(self):
        expected = {
            "route_to_debate",
            "route_to_execute",
            "route_to_hybrid",
            "route_rejected",
            "criteria_matched",
            "rule_matched",
            "default_route",
            "anomaly_detected",
        }
        actual = {e.value for e in RoutingEventType}
        assert actual == expected


# =============================================================================
# Data Class Tests
# =============================================================================


class TestRoutingCriteria:
    """Tests for RoutingCriteria dataclass."""

    def test_default_values(self):
        criteria = RoutingCriteria()
        assert criteria.financial_threshold == 10000.0
        assert RiskLevel.HIGH in criteria.risk_levels
        assert RiskLevel.CRITICAL in criteria.risk_levels
        assert "pii" in criteria.compliance_flags
        assert "hipaa" in criteria.compliance_flags
        assert criteria.stakeholder_threshold == 3
        assert criteria.time_sensitive_threshold_seconds == 60
        assert criteria.confidence_threshold == 0.85

    def test_custom_values(self):
        criteria = RoutingCriteria(
            financial_threshold=50000.0,
            risk_levels={RiskLevel.CRITICAL},
            compliance_flags={"gdpr"},
            stakeholder_threshold=5,
        )
        assert criteria.financial_threshold == 50000.0
        assert criteria.risk_levels == {RiskLevel.CRITICAL}
        assert criteria.compliance_flags == {"gdpr"}
        assert criteria.stakeholder_threshold == 5

    def test_risk_levels_normalize_strings(self):
        criteria = RoutingCriteria(risk_levels={"high", "critical"})
        assert RiskLevel.HIGH in criteria.risk_levels
        assert RiskLevel.CRITICAL in criteria.risk_levels

    def test_risk_levels_normalize_mixed(self):
        criteria = RoutingCriteria(risk_levels={RiskLevel.HIGH, "critical"})
        assert RiskLevel.HIGH in criteria.risk_levels
        assert RiskLevel.CRITICAL in criteria.risk_levels

    def test_risk_levels_unknown_string_preserved(self):
        criteria = RoutingCriteria(risk_levels={"unknown_level"})
        assert "unknown_level" in criteria.risk_levels

    def test_debate_keywords_default(self):
        criteria = RoutingCriteria()
        assert "consensus" in criteria.require_debate_keywords
        assert "debate" in criteria.require_debate_keywords
        assert "vote" in criteria.require_debate_keywords

    def test_execute_keywords_default(self):
        criteria = RoutingCriteria()
        assert "execute" in criteria.require_execute_keywords
        assert "run" in criteria.require_execute_keywords
        assert "just do it" in criteria.require_execute_keywords


class TestRouteDecision:
    """Tests for RouteDecision dataclass."""

    def test_basic_creation(self):
        decision = RouteDecision(
            destination=RouteDestination.DEBATE,
            reason="Test reason",
        )
        assert decision.destination == RouteDestination.DEBATE
        assert decision.reason == "Test reason"
        assert decision.criteria_matched == []
        assert decision.rule_id is None
        assert decision.confidence == 1.0
        assert decision.metadata == {}
        assert decision.decision_time_ms == 0.0
        assert decision.request_id == ""

    def test_full_creation(self):
        decision = RouteDecision(
            destination=RouteDestination.EXECUTE,
            reason="Direct execution",
            criteria_matched=["risk_level:low"],
            rule_id="rule-1",
            confidence=0.95,
            metadata={"key": "value"},
            decision_time_ms=1.5,
            request_id="req-001",
        )
        assert decision.rule_id == "rule-1"
        assert decision.confidence == 0.95
        assert decision.metadata == {"key": "value"}

    def test_to_dict(self):
        decision = RouteDecision(
            destination=RouteDestination.DEBATE,
            reason="Financial threshold exceeded",
            criteria_matched=["financial_threshold:50000>10000"],
            rule_id=None,
            confidence=0.95,
            decision_time_ms=2.3,
            request_id="req-002",
        )
        d = decision.to_dict()
        assert d["destination"] == "debate"
        assert d["reason"] == "Financial threshold exceeded"
        assert d["criteria_matched"] == ["financial_threshold:50000>10000"]
        assert d["rule_id"] is None
        assert d["confidence"] == 0.95
        assert d["decision_time_ms"] == 2.3
        assert d["request_id"] == "req-002"
        assert "timestamp" in d

    def test_timestamp_auto_generated(self):
        decision = RouteDecision(
            destination=RouteDestination.EXECUTE,
            reason="default",
        )
        assert isinstance(decision.timestamp, datetime)
        assert decision.timestamp.tzinfo == timezone.utc


class TestRoutingRule:
    """Tests for RoutingRule dataclass."""

    def test_basic_rule(self):
        rule = RoutingRule(
            rule_id="test-rule",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
        )
        assert rule.rule_id == "test-rule"
        assert rule.destination == RouteDestination.DEBATE
        assert rule.priority == 0
        assert rule.reason == ""
        assert rule.enabled is True
        assert rule.tenant_id is None
        assert rule.action_categories is None

    def test_full_rule(self):
        rule = RoutingRule(
            rule_id="full-rule",
            condition=lambda req: req.get("amount", 0) > 100,
            destination=RouteDestination.REJECT,
            priority=50,
            reason="Amount too large",
            enabled=False,
            tenant_id="tenant-1",
            action_categories={ActionCategory.FINANCIAL},
            metadata={"source": "test"},
        )
        assert rule.priority == 50
        assert rule.enabled is False
        assert rule.tenant_id == "tenant-1"
        assert ActionCategory.FINANCIAL in rule.action_categories

    def test_condition_is_callable(self):
        rule = RoutingRule(
            rule_id="callable",
            condition=lambda req: req.get("flag", False),
            destination=RouteDestination.EXECUTE,
        )
        assert rule.condition({"flag": True}) is True
        assert rule.condition({"flag": False}) is False
        assert rule.condition({}) is False


class TestTenantRoutingConfig:
    """Tests for TenantRoutingConfig dataclass."""

    def test_defaults(self):
        config = TenantRoutingConfig(tenant_id="acme")
        assert config.tenant_id == "acme"
        assert config.default_destination == RouteDestination.EXECUTE
        assert config.enabled_categories is None
        assert config.override_rules == []
        assert config.metadata == {}

    def test_custom_config(self):
        config = TenantRoutingConfig(
            tenant_id="corp",
            criteria=RoutingCriteria(financial_threshold=5000),
            default_destination=RouteDestination.DEBATE,
            enabled_categories={ActionCategory.FINANCIAL, ActionCategory.SECURITY},
        )
        assert config.criteria.financial_threshold == 5000
        assert config.default_destination == RouteDestination.DEBATE
        assert ActionCategory.FINANCIAL in config.enabled_categories


class TestCategoryDefaults:
    """Tests for CategoryDefaults dataclass."""

    def test_creation(self):
        cd = CategoryDefaults(
            category=ActionCategory.FINANCIAL,
            default_destination=RouteDestination.DEBATE,
            risk_level=RiskLevel.HIGH,
            requires_compliance_check=True,
        )
        assert cd.category == ActionCategory.FINANCIAL
        assert cd.default_destination == RouteDestination.DEBATE
        assert cd.risk_level == RiskLevel.HIGH
        assert cd.requires_compliance_check is True

    def test_defaults(self):
        cd = CategoryDefaults(
            category=ActionCategory.GENERAL,
            default_destination=RouteDestination.EXECUTE,
        )
        assert cd.risk_level == RiskLevel.LOW
        assert cd.requires_compliance_check is False


class TestRoutingMetrics:
    """Tests for RoutingMetrics dataclass."""

    def test_defaults(self):
        m = RoutingMetrics()
        assert m.total_requests == 0
        assert m.debate_routes == 0
        assert m.execute_routes == 0
        assert m.hybrid_routes == 0
        assert m.rejected_routes == 0
        assert m.avg_decision_time_ms == 0.0

    def test_to_dict(self):
        m = RoutingMetrics(
            total_requests=10,
            debate_routes=3,
            execute_routes=5,
            hybrid_routes=1,
            rejected_routes=1,
            avg_decision_time_ms=2.5,
        )
        d = m.to_dict()
        assert d["total_requests"] == 10
        assert d["debate_ratio"] == 0.3
        assert d["execute_ratio"] == 0.5
        assert d["avg_decision_time_ms"] == 2.5

    def test_to_dict_zero_requests_no_division_error(self):
        m = RoutingMetrics()
        d = m.to_dict()
        # Should use 1 as divisor to avoid ZeroDivisionError
        assert d["debate_ratio"] == 0.0
        assert d["execute_ratio"] == 0.0

    def test_criteria_matches_default_factory(self):
        m = RoutingMetrics()
        m.criteria_matches["test"] += 1
        assert m.criteria_matches["test"] == 1


class TestRoutingAuditEntry:
    """Tests for RoutingAuditEntry dataclass."""

    def test_creation(self):
        ts = datetime.now(timezone.utc)
        entry = RoutingAuditEntry(
            timestamp=ts,
            event_type=RoutingEventType.ROUTE_TO_DEBATE,
            request_id="req-1",
            tenant_id="tenant-1",
        )
        assert entry.timestamp == ts
        assert entry.event_type == RoutingEventType.ROUTE_TO_DEBATE
        assert entry.request_id == "req-1"
        assert entry.tenant_id == "tenant-1"
        assert entry.decision is None
        assert entry.metadata == {}


# =============================================================================
# DEFAULT_CATEGORY_CONFIGS Tests
# =============================================================================


class TestDefaultCategoryConfigs:
    """Tests for DEFAULT_CATEGORY_CONFIGS."""

    def test_all_categories_have_config(self):
        for cat in ActionCategory:
            assert cat in DEFAULT_CATEGORY_CONFIGS

    def test_financial_routes_to_debate(self):
        cfg = DEFAULT_CATEGORY_CONFIGS[ActionCategory.FINANCIAL]
        assert cfg.default_destination == RouteDestination.DEBATE
        assert cfg.risk_level == RiskLevel.HIGH
        assert cfg.requires_compliance_check is True

    def test_compliance_routes_to_debate(self):
        cfg = DEFAULT_CATEGORY_CONFIGS[ActionCategory.COMPLIANCE]
        assert cfg.default_destination == RouteDestination.DEBATE

    def test_security_routes_to_debate(self):
        cfg = DEFAULT_CATEGORY_CONFIGS[ActionCategory.SECURITY]
        assert cfg.default_destination == RouteDestination.DEBATE
        assert cfg.risk_level == RiskLevel.CRITICAL

    def test_infrastructure_routes_to_hybrid(self):
        cfg = DEFAULT_CATEGORY_CONFIGS[ActionCategory.INFRASTRUCTURE]
        assert cfg.default_destination == RouteDestination.HYBRID_DEBATE_THEN_EXECUTE

    def test_communication_routes_to_execute(self):
        cfg = DEFAULT_CATEGORY_CONFIGS[ActionCategory.COMMUNICATION]
        assert cfg.default_destination == RouteDestination.EXECUTE
        assert cfg.risk_level == RiskLevel.LOW

    def test_analytics_routes_to_execute(self):
        cfg = DEFAULT_CATEGORY_CONFIGS[ActionCategory.ANALYTICS]
        assert cfg.default_destination == RouteDestination.EXECUTE

    def test_general_routes_to_execute(self):
        cfg = DEFAULT_CATEGORY_CONFIGS[ActionCategory.GENERAL]
        assert cfg.default_destination == RouteDestination.EXECUTE


# =============================================================================
# DecisionRouter Tests - Initialization
# =============================================================================


class TestDecisionRouterInit:
    """Tests for DecisionRouter initialization."""

    def test_default_init(self):
        router = DecisionRouter()
        assert router._default_destination == RouteDestination.EXECUTE
        assert router._enable_audit is True
        assert router._max_audit_entries == 10000
        assert router._request_counter == 0

    def test_custom_criteria(self):
        criteria = RoutingCriteria(financial_threshold=50000)
        router = DecisionRouter(criteria=criteria)
        assert router._criteria.financial_threshold == 50000

    def test_custom_default_destination(self):
        router = DecisionRouter(default_destination=RouteDestination.DEBATE)
        assert router._default_destination == RouteDestination.DEBATE

    def test_disable_audit(self):
        router = DecisionRouter(enable_audit=False)
        assert router._enable_audit is False

    def test_custom_category_configs(self):
        custom = {
            ActionCategory.GENERAL: CategoryDefaults(
                category=ActionCategory.GENERAL,
                default_destination=RouteDestination.DEBATE,
            )
        }
        router = DecisionRouter(category_configs=custom)
        assert (
            router._category_configs[ActionCategory.GENERAL].default_destination
            == RouteDestination.DEBATE
        )
        # Other categories should still use defaults
        assert (
            router._category_configs[ActionCategory.FINANCIAL].default_destination
            == RouteDestination.DEBATE
        )


# =============================================================================
# DecisionRouter Tests - Rule Management
# =============================================================================


class TestRuleManagement:
    """Tests for DecisionRouter rule management."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    def test_add_rule(self, router):
        rule = RoutingRule(
            rule_id="r1",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
        )
        router.add_rule(rule)
        assert router.get_rule("r1") is rule

    def test_remove_rule_exists(self, router):
        rule = RoutingRule(
            rule_id="r1",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
        )
        router.add_rule(rule)
        assert router.remove_rule("r1") is True
        assert router.get_rule("r1") is None

    def test_remove_rule_not_found(self, router):
        assert router.remove_rule("nonexistent") is False

    def test_get_rule_not_found(self, router):
        assert router.get_rule("nonexistent") is None

    def test_list_rules_sorted_by_priority(self, router):
        r1 = RoutingRule(
            rule_id="r1",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
            priority=10,
        )
        r2 = RoutingRule(
            rule_id="r2",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
            priority=50,
        )
        r3 = RoutingRule(
            rule_id="r3",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
            priority=30,
        )
        router.add_rule(r1)
        router.add_rule(r2)
        router.add_rule(r3)
        rules = router.list_rules()
        assert [r.rule_id for r in rules] == ["r2", "r3", "r1"]

    def test_enable_rule(self, router):
        rule = RoutingRule(
            rule_id="r1",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
            enabled=False,
        )
        router.add_rule(rule)
        assert router.enable_rule("r1") is True
        assert router.get_rule("r1").enabled is True

    def test_enable_rule_not_found(self, router):
        assert router.enable_rule("nonexistent") is False

    def test_disable_rule(self, router):
        rule = RoutingRule(
            rule_id="r1",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
        )
        router.add_rule(rule)
        assert router.disable_rule("r1") is True
        assert router.get_rule("r1").enabled is False

    def test_disable_rule_not_found(self, router):
        assert router.disable_rule("nonexistent") is False

    def test_overwrite_rule_same_id(self, router):
        rule1 = RoutingRule(
            rule_id="r1", condition=lambda req: True, destination=RouteDestination.DEBATE
        )
        rule2 = RoutingRule(
            rule_id="r1", condition=lambda req: False, destination=RouteDestination.EXECUTE
        )
        router.add_rule(rule1)
        router.add_rule(rule2)
        assert router.get_rule("r1").destination == RouteDestination.EXECUTE


# =============================================================================
# DecisionRouter Tests - Tenant Configuration
# =============================================================================


class TestTenantConfiguration:
    """Tests for DecisionRouter tenant configuration."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_add_tenant_config(self, router):
        config = TenantRoutingConfig(tenant_id="acme")
        await router.add_tenant_config(config)
        assert router.get_tenant_config("acme") is config

    @pytest.mark.asyncio
    async def test_remove_tenant_config(self, router):
        config = TenantRoutingConfig(tenant_id="acme")
        await router.add_tenant_config(config)
        assert await router.remove_tenant_config("acme") is True
        assert router.get_tenant_config("acme") is None

    @pytest.mark.asyncio
    async def test_remove_tenant_config_not_found(self, router):
        assert await router.remove_tenant_config("nonexistent") is False

    def test_get_tenant_config_not_found(self, router):
        assert router.get_tenant_config("missing") is None

    @pytest.mark.asyncio
    async def test_list_tenant_configs(self, router):
        c1 = TenantRoutingConfig(tenant_id="t1")
        c2 = TenantRoutingConfig(tenant_id="t2")
        await router.add_tenant_config(c1)
        await router.add_tenant_config(c2)
        configs = router.list_tenant_configs()
        assert len(configs) == 2
        ids = {c.tenant_id for c in configs}
        assert ids == {"t1", "t2"}

    @pytest.mark.asyncio
    async def test_update_tenant_config(self, router):
        c1 = TenantRoutingConfig(
            tenant_id="acme",
            default_destination=RouteDestination.EXECUTE,
        )
        await router.add_tenant_config(c1)
        c2 = TenantRoutingConfig(
            tenant_id="acme",
            default_destination=RouteDestination.DEBATE,
        )
        await router.add_tenant_config(c2)
        assert router.get_tenant_config("acme").default_destination == RouteDestination.DEBATE


# =============================================================================
# DecisionRouter Tests - Default Routing
# =============================================================================


class TestDefaultRouting:
    """Tests for default routing behavior (no criteria matched)."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_default_routes_to_execute(self, router):
        decision = await router.route({"action": "something_generic"})
        # "something_generic" -> GENERAL category -> EXECUTE
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_default_with_custom_destination(self):
        router = DecisionRouter(default_destination=RouteDestination.DEBATE)
        # An action with no category or other matching criteria that has
        # no inferred category keyword should fall through to GENERAL category default.
        # Since GENERAL defaults to EXECUTE and category defaults take precedence
        # over global default, we need a request that yields no category at all.
        # The _extract_action_category always returns GENERAL as fallback, so
        # category config will always apply. Let's test with empty category configs.
        router._category_configs.clear()
        decision = await router.route({})
        assert decision.destination == RouteDestination.DEBATE
        assert "global_default" in decision.criteria_matched

    @pytest.mark.asyncio
    async def test_general_category_default(self, router):
        decision = await router.route({"action": "something"})
        assert decision.destination == RouteDestination.EXECUTE
        assert any("category_default:general" in c for c in decision.criteria_matched)


# =============================================================================
# DecisionRouter Tests - Financial Threshold Routing
# =============================================================================


class TestFinancialThresholdRouting:
    """Tests for financial threshold-based routing."""

    @pytest.fixture
    def router(self):
        return DecisionRouter(criteria=RoutingCriteria(financial_threshold=10000.0))

    @pytest.mark.asyncio
    async def test_amount_above_threshold_routes_to_debate(self, router):
        decision = await router.route({"amount": 50000})
        assert decision.destination == RouteDestination.DEBATE
        assert any("financial_threshold" in c for c in decision.criteria_matched)

    @pytest.mark.asyncio
    async def test_amount_at_threshold_routes_to_default(self, router):
        decision = await router.route({"amount": 10000})
        # Equal to threshold, not above -> does not trigger
        assert decision.destination != RouteDestination.REJECT  # goes to default

    @pytest.mark.asyncio
    async def test_amount_below_threshold_routes_to_default(self, router):
        decision = await router.route({"amount": 5000})
        # Below threshold -> falls through to default category routing
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_amount_just_above_threshold(self, router):
        decision = await router.route({"amount": 10000.01})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_string_amount_converted(self, router):
        decision = await router.route({"amount": "50000"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_invalid_amount_ignored(self, router):
        decision = await router.route({"amount": "not_a_number"})
        # Invalid amount treated as 0 -> falls through
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_no_amount_field(self, router):
        decision = await router.route({"action": "something"})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_zero_amount(self, router):
        decision = await router.route({"amount": 0})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_negative_amount(self, router):
        decision = await router.route({"amount": -5000})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_decision_metadata_includes_amount(self, router):
        decision = await router.route({"amount": 15000})
        assert decision.metadata.get("amount") == 15000
        assert decision.metadata.get("threshold") == 10000.0


# =============================================================================
# DecisionRouter Tests - Risk Level Routing
# =============================================================================


class TestRiskLevelRouting:
    """Tests for risk level-based routing."""

    @pytest.fixture
    def router(self):
        return DecisionRouter(
            criteria=RoutingCriteria(
                risk_levels={RiskLevel.HIGH, RiskLevel.CRITICAL},
            )
        )

    @pytest.mark.asyncio
    async def test_high_risk_routes_to_debate(self, router):
        decision = await router.route({"risk_level": "high"})
        assert decision.destination == RouteDestination.DEBATE
        assert any("risk_level:high" in c for c in decision.criteria_matched)

    @pytest.mark.asyncio
    async def test_critical_risk_routes_to_debate(self, router):
        decision = await router.route({"risk_level": "critical"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_low_risk_not_triggered(self, router):
        decision = await router.route({"risk_level": "low"})
        # Low risk does not trigger debate; goes to default
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_medium_risk_not_triggered(self, router):
        decision = await router.route({"risk_level": "medium"})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_risk_key_alias(self, router):
        """Supports 'risk' as alternative key."""
        decision = await router.route({"risk": "high"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_invalid_risk_level_ignored(self, router):
        decision = await router.route({"risk_level": "unknown"})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_risk_level_case_insensitive(self, router):
        decision = await router.route({"risk_level": "HIGH"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_risk_metadata(self, router):
        decision = await router.route({"risk_level": "critical"})
        assert decision.metadata.get("risk_level") == "critical"


# =============================================================================
# DecisionRouter Tests - Compliance Flag Routing
# =============================================================================


class TestComplianceFlagRouting:
    """Tests for compliance flag-based routing."""

    @pytest.fixture
    def router(self):
        return DecisionRouter(
            criteria=RoutingCriteria(
                compliance_flags={"pii", "hipaa", "gdpr", "sox", "financial"},
            )
        )

    @pytest.mark.asyncio
    async def test_pii_flag_routes_to_debate(self, router):
        decision = await router.route({"compliance_flags": ["pii"]})
        assert decision.destination == RouteDestination.DEBATE
        assert any("compliance_flags" in c for c in decision.criteria_matched)

    @pytest.mark.asyncio
    async def test_hipaa_flag_routes_to_debate(self, router):
        decision = await router.route({"compliance_flags": ["hipaa"]})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_multiple_flags(self, router):
        decision = await router.route({"compliance_flags": ["pii", "gdpr"]})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_no_matching_flags(self, router):
        decision = await router.route({"compliance_flags": ["other"]})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_string_flag_treated_as_single(self, router):
        decision = await router.route({"compliance_flags": "pii"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_flags_via_tags(self, router):
        decision = await router.route({"tags": ["hipaa", "report"]})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_flags_via_labels(self, router):
        decision = await router.route({"labels": ["gdpr"]})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_flags_case_insensitive(self, router):
        decision = await router.route({"compliance_flags": ["PII"]})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_compliance_metadata(self, router):
        decision = await router.route({"compliance_flags": ["pii", "hipaa"]})
        assert "pii" in decision.metadata.get("compliance_flags", [])
        assert "hipaa" in decision.metadata.get("compliance_flags", [])


# =============================================================================
# DecisionRouter Tests - Keyword-Based Routing
# =============================================================================


class TestKeywordRouting:
    """Tests for explicit user intent via keywords."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_debate_keyword_in_content(self, router):
        decision = await router.route({"content": "We need to debate this approach"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_consensus_keyword(self, router):
        decision = await router.route({"content": "Seek consensus on the proposal"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_discuss_keyword(self, router):
        decision = await router.route({"content": "Let's discuss the options"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_vote_keyword(self, router):
        decision = await router.route({"content": "Put it to a vote"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_approve_keyword(self, router):
        decision = await router.route({"content": "We need to approve this"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_execute_keyword_in_content(self, router):
        decision = await router.route({"content": "Execute the deployment now"})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_run_keyword(self, router):
        decision = await router.route({"content": "Run the migration"})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_just_do_it_keyword(self, router):
        decision = await router.route({"content": "just do it already"})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_keyword_in_description_field(self, router):
        decision = await router.route({"description": "Let us debate the tradeoffs"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_debate_keyword_takes_priority(self, router):
        """Debate keywords checked before execute keywords."""
        decision = await router.route({"content": "debate whether to execute"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_no_keywords_matched(self, router):
        decision = await router.route({"content": "Process the data"})
        # No keyword matches, falls through to default category routing
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_keyword_case_insensitive(self, router):
        decision = await router.route({"content": "DEBATE the strategy"})
        assert decision.destination == RouteDestination.DEBATE


# =============================================================================
# DecisionRouter Tests - Explicit Routing Request
# =============================================================================


class TestExplicitRoutingRequest:
    """Tests for explicit route_to / destination fields."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_route_to_debate(self, router):
        decision = await router.route({"route_to": "debate"})
        assert decision.destination == RouteDestination.DEBATE
        assert decision.confidence == 1.0

    @pytest.mark.asyncio
    async def test_route_to_execute(self, router):
        decision = await router.route({"route_to": "execute"})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_destination_field(self, router):
        decision = await router.route({"destination": "reject"})
        assert decision.destination == RouteDestination.REJECT

    @pytest.mark.asyncio
    async def test_invalid_destination_ignored(self, router):
        decision = await router.route({"route_to": "invalid_dest"})
        # Invalid value -> ignored, falls through to defaults
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_route_to_hybrid(self, router):
        decision = await router.route({"route_to": "hybrid_debate_execute"})
        assert decision.destination == RouteDestination.HYBRID_DEBATE_THEN_EXECUTE


# =============================================================================
# DecisionRouter Tests - Custom Rule Evaluation
# =============================================================================


class TestCustomRuleEvaluation:
    """Tests for custom rule evaluation during routing."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_matching_rule_routes(self, router):
        rule = RoutingRule(
            rule_id="big-tx",
            condition=lambda req: req.get("amount", 0) > 100000,
            destination=RouteDestination.DEBATE,
            reason="Very large transaction",
            priority=100,
        )
        router.add_rule(rule)
        decision = await router.route({"amount": 200000})
        assert decision.destination == RouteDestination.DEBATE
        assert decision.rule_id == "big-tx"
        assert "rule:big-tx" in decision.criteria_matched

    @pytest.mark.asyncio
    async def test_non_matching_rule_skipped(self, router):
        rule = RoutingRule(
            rule_id="impossible",
            condition=lambda req: req.get("secret_flag"),
            destination=RouteDestination.REJECT,
            priority=100,
        )
        router.add_rule(rule)
        decision = await router.route({"action": "normal"})
        assert decision.destination == RouteDestination.EXECUTE
        assert decision.rule_id is None

    @pytest.mark.asyncio
    async def test_disabled_rule_skipped(self, router):
        rule = RoutingRule(
            rule_id="disabled",
            condition=lambda req: True,
            destination=RouteDestination.REJECT,
            enabled=False,
        )
        router.add_rule(rule)
        decision = await router.route({"action": "anything"})
        assert decision.destination != RouteDestination.REJECT

    @pytest.mark.asyncio
    async def test_higher_priority_rule_wins(self, router):
        low = RoutingRule(
            rule_id="low",
            condition=lambda req: True,
            destination=RouteDestination.EXECUTE,
            priority=10,
        )
        high = RoutingRule(
            rule_id="high",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
            priority=100,
        )
        router.add_rule(low)
        router.add_rule(high)
        decision = await router.route({})
        assert decision.destination == RouteDestination.DEBATE
        assert decision.rule_id == "high"

    @pytest.mark.asyncio
    async def test_rule_with_exception_skipped(self, router):
        """Rules that raise exceptions are skipped gracefully."""

        def bad_condition(req):
            raise RuntimeError("rule error")

        bad_rule = RoutingRule(
            rule_id="bad",
            condition=bad_condition,
            destination=RouteDestination.REJECT,
            priority=100,
        )
        router.add_rule(bad_rule)
        decision = await router.route({"action": "test"})
        # Should not crash, should route to default
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_tenant_restricted_rule(self, router):
        rule = RoutingRule(
            rule_id="tenant-only",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
            tenant_id="acme",
            priority=100,
        )
        router.add_rule(rule)
        # Request from different tenant -> rule skipped
        decision = await router.route({}, context={"tenant_id": "other"})
        assert decision.destination == RouteDestination.EXECUTE

        # Request from matching tenant -> rule matches
        decision = await router.route({}, context={"tenant_id": "acme"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_category_restricted_rule(self, router):
        rule = RoutingRule(
            rule_id="fin-only",
            condition=lambda req: True,
            destination=RouteDestination.REJECT,
            action_categories={ActionCategory.FINANCIAL},
            priority=100,
        )
        router.add_rule(rule)
        # Financial action -> rule matches
        decision = await router.route({"action": "transfer_funds"})
        assert decision.destination == RouteDestination.REJECT

        # Non-financial action -> rule skipped (category mismatch)
        decision = await router.route({"action": "generic_task"})
        assert decision.destination != RouteDestination.REJECT

    @pytest.mark.asyncio
    async def test_rule_metadata_propagated(self, router):
        rule = RoutingRule(
            rule_id="meta",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
            metadata={"source": "custom", "version": 2},
            priority=100,
        )
        router.add_rule(rule)
        decision = await router.route({})
        assert decision.metadata.get("source") == "custom"
        assert decision.metadata.get("version") == 2


# =============================================================================
# DecisionRouter Tests - Stakeholder Routing
# =============================================================================


class TestStakeholderRouting:
    """Tests for stakeholder count-based routing."""

    @pytest.fixture
    def router(self):
        return DecisionRouter(
            criteria=RoutingCriteria(stakeholder_threshold=3),
        )

    @pytest.mark.asyncio
    async def test_many_stakeholders_route_to_debate(self, router):
        decision = await router.route({"stakeholders": ["a", "b", "c"]})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_few_stakeholders_no_trigger(self, router):
        decision = await router.route({"stakeholders": ["a", "b"]})
        # Only 2 stakeholders, threshold is 3
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_stakeholders_plus_approvers(self, router):
        decision = await router.route(
            {
                "stakeholders": ["a"],
                "approvers": ["b", "c"],
            }
        )
        # 1 stakeholder + 2 approvers = 3 >= threshold
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_stakeholders_plus_reviewers(self, router):
        decision = await router.route(
            {
                "stakeholders": ["a"],
                "reviewers": ["b", "c"],
            }
        )
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_comma_separated_stakeholders(self, router):
        decision = await router.route({"stakeholders": "alice,bob,charlie"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_approvers_as_single_string(self, router):
        decision = await router.route(
            {
                "stakeholders": ["a", "b"],
                "approvers": "c",  # single string -> count as 1
            }
        )
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_stakeholder_metadata(self, router):
        decision = await router.route({"stakeholders": ["a", "b", "c", "d"]})
        assert decision.metadata.get("stakeholder_count") == 4


# =============================================================================
# DecisionRouter Tests - Per-Tenant Routing
# =============================================================================


class TestPerTenantRouting:
    """Tests for per-tenant routing configuration."""

    @pytest.fixture
    def router(self):
        return DecisionRouter(
            criteria=RoutingCriteria(financial_threshold=10000),
        )

    @pytest.mark.asyncio
    async def test_tenant_overrides_global_criteria(self, router):
        tenant_config = TenantRoutingConfig(
            tenant_id="acme",
            criteria=RoutingCriteria(financial_threshold=1000),
        )
        await router.add_tenant_config(tenant_config)

        # Global threshold is 10000, but tenant has 1000
        decision = await router.route(
            {"amount": 5000},
            context={"tenant_id": "acme"},
        )
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_global_criteria_for_unknown_tenant(self, router):
        decision = await router.route(
            {"amount": 5000},
            context={"tenant_id": "unknown"},
        )
        # 5000 < global threshold 10000 -> falls through
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_tenant_default_destination(self, router):
        tenant_config = TenantRoutingConfig(
            tenant_id="corp",
            default_destination=RouteDestination.DEBATE,
        )
        await router.add_tenant_config(tenant_config)

        # Clear category configs so we hit the tenant default
        router._category_configs.clear()

        decision = await router.route({}, context={"tenant_id": "corp"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_tenant_override_rules(self, router):
        override_rule = RoutingRule(
            rule_id="tenant-emergency",
            condition=lambda req: req.get("emergency", False),
            destination=RouteDestination.REJECT,
            priority=200,
            reason="Emergency override",
        )
        tenant_config = TenantRoutingConfig(
            tenant_id="acme",
            override_rules=[override_rule],
        )
        await router.add_tenant_config(tenant_config)

        decision = await router.route(
            {"emergency": True},
            context={"tenant_id": "acme"},
        )
        assert decision.destination == RouteDestination.REJECT
        assert decision.rule_id == "tenant-emergency"


# =============================================================================
# DecisionRouter Tests - Category Default Routing
# =============================================================================


class TestCategoryDefaultRouting:
    """Tests for action category-based default routing."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_financial_action_inferred(self, router):
        decision = await router.route({"action": "transfer_funds"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_security_action_inferred(self, router):
        decision = await router.route({"action": "update_security_settings"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_infrastructure_action_inferred(self, router):
        decision = await router.route({"action": "deploy_service"})
        assert decision.destination == RouteDestination.HYBRID_DEBATE_THEN_EXECUTE

    @pytest.mark.asyncio
    async def test_data_management_action_inferred(self, router):
        decision = await router.route({"action": "backup_data"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_user_management_action_inferred(self, router):
        decision = await router.route({"action": "create_user_account"})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_compliance_action_inferred(self, router):
        decision = await router.route({"action": "audit_review"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_explicit_category_field(self, router):
        decision = await router.route({"category": "security"})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_action_category_field(self, router):
        decision = await router.route({"action_category": "analytics"})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_explicit_category_enum(self, router):
        decision = await router.route({"category": ActionCategory.FINANCIAL})
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_update_category_config(self, router):
        router.update_category_config(
            ActionCategory.GENERAL,
            CategoryDefaults(
                category=ActionCategory.GENERAL,
                default_destination=RouteDestination.DEBATE,
            ),
        )
        decision = await router.route({"action": "generic_thing"})
        assert decision.destination == RouteDestination.DEBATE


# =============================================================================
# DecisionRouter Tests - Hybrid Routing
# =============================================================================


class TestHybridRouting:
    """Tests for hybrid routing modes."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_hybrid_debate_then_execute_via_explicit(self, router):
        decision = await router.route({"route_to": "hybrid_debate_execute"})
        assert decision.destination == RouteDestination.HYBRID_DEBATE_THEN_EXECUTE

    @pytest.mark.asyncio
    async def test_hybrid_execute_with_validation_via_explicit(self, router):
        decision = await router.route({"route_to": "hybrid_execute_validate"})
        assert decision.destination == RouteDestination.HYBRID_EXECUTE_WITH_VALIDATION

    @pytest.mark.asyncio
    async def test_hybrid_via_rule(self, router):
        rule = RoutingRule(
            rule_id="hybrid-rule",
            condition=lambda req: req.get("needs_validation", False),
            destination=RouteDestination.HYBRID_EXECUTE_WITH_VALIDATION,
            priority=100,
        )
        router.add_rule(rule)
        decision = await router.route({"needs_validation": True})
        assert decision.destination == RouteDestination.HYBRID_EXECUTE_WITH_VALIDATION

    @pytest.mark.asyncio
    async def test_infrastructure_defaults_to_hybrid(self, router):
        decision = await router.route({"action": "deploy_new_server"})
        assert decision.destination == RouteDestination.HYBRID_DEBATE_THEN_EXECUTE


# =============================================================================
# DecisionRouter Tests - Metrics Tracking
# =============================================================================


class TestMetricsTracking:
    """Tests for routing metrics tracking."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_total_requests_increments(self, router):
        await router.route({"action": "test1"})
        await router.route({"action": "test2"})
        metrics = router.get_metrics()
        assert metrics["total_requests"] == 2

    @pytest.mark.asyncio
    async def test_debate_count(self, router):
        await router.route({"risk_level": "high"})
        await router.route({"risk_level": "critical"})
        metrics = router.get_metrics()
        assert metrics["debate_routes"] == 2

    @pytest.mark.asyncio
    async def test_execute_count(self, router):
        await router.route({"action": "generic"})
        metrics = router.get_metrics()
        assert metrics["execute_routes"] == 1

    @pytest.mark.asyncio
    async def test_hybrid_count(self, router):
        await router.route({"action": "deploy_server"})
        metrics = router.get_metrics()
        assert metrics["hybrid_routes"] == 1

    @pytest.mark.asyncio
    async def test_rejected_count(self, router):
        rule = RoutingRule(
            rule_id="reject-all",
            condition=lambda req: True,
            destination=RouteDestination.REJECT,
            priority=999,
        )
        router.add_rule(rule)
        await router.route({})
        metrics = router.get_metrics()
        assert metrics["rejected_routes"] == 1

    @pytest.mark.asyncio
    async def test_debate_ratio(self, router):
        await router.route({"risk_level": "high"})
        await router.route({"action": "generic"})
        metrics = router.get_metrics()
        assert metrics["debate_ratio"] == 0.5
        assert metrics["execute_ratio"] == 0.5

    @pytest.mark.asyncio
    async def test_criteria_matches_tracked(self, router):
        await router.route({"risk_level": "high"})
        metrics = router.get_metrics()
        assert any("risk_level" in k for k in metrics["criteria_matches"])

    @pytest.mark.asyncio
    async def test_rule_matches_tracked(self, router):
        rule = RoutingRule(
            rule_id="tracked",
            condition=lambda req: True,
            destination=RouteDestination.DEBATE,
            priority=100,
        )
        router.add_rule(rule)
        await router.route({})
        await router.route({})
        metrics = router.get_metrics()
        assert metrics["rule_matches"].get("tracked") == 2

    @pytest.mark.asyncio
    async def test_decision_time_recorded(self, router):
        await router.route({})
        metrics = router.get_metrics()
        assert metrics["avg_decision_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_tenant_metrics(self, router):
        await router.route({"action": "generic"}, context={"tenant_id": "acme"})
        await router.route({"action": "generic"}, context={"tenant_id": "acme"})
        tenant_metrics = router.get_tenant_metrics("acme")
        assert tenant_metrics is not None
        assert tenant_metrics["total_requests"] == 2

    @pytest.mark.asyncio
    async def test_tenant_metrics_not_found(self, router):
        assert router.get_tenant_metrics("nonexistent") is None

    @pytest.mark.asyncio
    async def test_all_tenant_metrics(self, router):
        await router.route({}, context={"tenant_id": "t1"})
        await router.route({}, context={"tenant_id": "t2"})
        all_metrics = router.get_all_tenant_metrics()
        assert "t1" in all_metrics
        assert "t2" in all_metrics

    @pytest.mark.asyncio
    async def test_debate_vs_execute_ratio_global(self, router):
        await router.route({"risk_level": "high"})
        await router.route({"action": "generic"})
        ratio = router.get_debate_vs_execute_ratio()
        assert ratio["debate_ratio"] == 0.5
        assert ratio["execute_ratio"] == 0.5

    @pytest.mark.asyncio
    async def test_debate_vs_execute_ratio_tenant(self, router):
        await router.route({"risk_level": "high"}, context={"tenant_id": "acme"})
        ratio = router.get_debate_vs_execute_ratio("acme")
        assert ratio["debate_ratio"] == 1.0

    @pytest.mark.asyncio
    async def test_debate_vs_execute_ratio_unknown_tenant(self, router):
        ratio = router.get_debate_vs_execute_ratio("unknown")
        assert ratio["debate_ratio"] == 0.0


# =============================================================================
# DecisionRouter Tests - Anomaly Detection
# =============================================================================


class TestAnomalyDetection:
    """Tests for anomaly detection during routing."""

    @pytest.mark.asyncio
    async def test_anomaly_detector_called(self):
        alerts_sent: list[dict[str, Any]] = []

        class MockAlertHandler:
            async def send_alert(self, alert_type, message, severity, metadata):
                alerts_sent.append(
                    {
                        "type": alert_type,
                        "message": message,
                        "severity": severity,
                        "metadata": metadata,
                    }
                )

        detector = SimpleAnomalyDetector(
            debate_ratio_high_threshold=0.8,
            min_samples=2,
        )
        router = DecisionRouter(
            anomaly_detector=detector,
            alert_handler=MockAlertHandler(),
        )

        # Force debate_routes and total_requests to trigger anomaly
        router._metrics.total_requests = 10
        router._metrics.debate_routes = 9

        await router.route({"risk_level": "high"})

        # The anomaly should have triggered an alert
        assert len(alerts_sent) >= 1
        assert alerts_sent[0]["type"] == "routing_anomaly"

    @pytest.mark.asyncio
    async def test_anomaly_audit_entry_created(self):
        detector = SimpleAnomalyDetector(
            debate_ratio_high_threshold=0.5,
            min_samples=2,
        )
        router = DecisionRouter(anomaly_detector=detector)

        # Set up state to trigger anomaly
        router._metrics.total_requests = 10
        router._metrics.debate_routes = 9

        await router.route({"risk_level": "high"})

        audit_log = await router.get_audit_log()
        anomaly_entries = [e for e in audit_log if e.get("event_type") == "anomaly_detected"]
        assert len(anomaly_entries) >= 1

    @pytest.mark.asyncio
    async def test_no_anomaly_below_min_samples(self):
        detector = SimpleAnomalyDetector(min_samples=1000)
        router = DecisionRouter(anomaly_detector=detector)

        await router.route({"risk_level": "high"})

        audit_log = await router.get_audit_log()
        anomaly_entries = [e for e in audit_log if e.get("event_type") == "anomaly_detected"]
        assert len(anomaly_entries) == 0

    @pytest.mark.asyncio
    async def test_alert_handler_exception_handled(self):
        class FailingAlertHandler:
            async def send_alert(self, alert_type, message, severity, metadata):
                raise RuntimeError("Alert service down")

        detector = SimpleAnomalyDetector(
            debate_ratio_high_threshold=0.5,
            min_samples=2,
        )
        router = DecisionRouter(
            anomaly_detector=detector,
            alert_handler=FailingAlertHandler(),
        )
        router._metrics.total_requests = 10
        router._metrics.debate_routes = 9

        # Should not raise
        decision = await router.route({"risk_level": "high"})
        assert decision is not None


# =============================================================================
# DecisionRouter Tests - Audit Logging
# =============================================================================


class TestAuditLogging:
    """Tests for audit log functionality."""

    @pytest.fixture
    def router(self):
        return DecisionRouter(enable_audit=True)

    @pytest.mark.asyncio
    async def test_audit_entries_created(self, router):
        await router.route({"action": "test"})
        log = await router.get_audit_log()
        assert len(log) >= 1

    @pytest.mark.asyncio
    async def test_audit_entry_fields(self, router):
        await router.route({"action": "test"})
        log = await router.get_audit_log()
        entry = log[0]
        assert "timestamp" in entry
        assert "event_type" in entry
        assert "request_id" in entry
        assert "decision" in entry

    @pytest.mark.asyncio
    async def test_audit_filter_by_tenant(self, router):
        await router.route({}, context={"tenant_id": "t1"})
        await router.route({}, context={"tenant_id": "t2"})
        log = await router.get_audit_log(tenant_id="t1")
        assert all(e["tenant_id"] == "t1" for e in log)

    @pytest.mark.asyncio
    async def test_audit_filter_by_event_type(self, router):
        await router.route({"risk_level": "high"})
        await router.route({"action": "generic"})
        log = await router.get_audit_log(event_type=RoutingEventType.CRITERIA_MATCHED)
        assert all(e["event_type"] == "criteria_matched" for e in log)

    @pytest.mark.asyncio
    async def test_audit_filter_by_since(self, router):
        before = datetime.now(timezone.utc)
        await router.route({})
        log = await router.get_audit_log(since=before)
        assert len(log) >= 1

    @pytest.mark.asyncio
    async def test_audit_limit(self, router):
        for _ in range(10):
            await router.route({})
        log = await router.get_audit_log(limit=3)
        assert len(log) == 3

    @pytest.mark.asyncio
    async def test_audit_disabled(self):
        router = DecisionRouter(enable_audit=False)
        await router.route({})
        log = await router.get_audit_log()
        assert len(log) == 0

    @pytest.mark.asyncio
    async def test_audit_max_entries_enforced(self):
        router = DecisionRouter(max_audit_entries=5)
        for _ in range(10):
            await router.route({})
        # Internal log should be capped at 5
        assert len(router._audit_log) <= 5


# =============================================================================
# DecisionRouter Tests - Event Handlers
# =============================================================================


class TestEventHandlers:
    """Tests for event handler registration and notification."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_sync_event_handler_called(self, router):
        events: list[RoutingAuditEntry] = []
        router.add_event_handler(lambda entry: events.append(entry))
        await router.route({"action": "test"})
        assert len(events) == 1
        assert isinstance(events[0], RoutingAuditEntry)

    @pytest.mark.asyncio
    async def test_async_event_handler_called(self, router):
        events: list[RoutingAuditEntry] = []

        async def handler(entry):
            events.append(entry)

        router.add_event_handler(handler)
        await router.route({"action": "test"})
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_remove_event_handler(self, router):
        events: list[RoutingAuditEntry] = []

        def handler(entry: RoutingAuditEntry) -> None:
            events.append(entry)

        router.add_event_handler(handler)
        router.remove_event_handler(handler)
        await router.route({"action": "test"})
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_handler_safe(self, router):
        # Should not raise
        router.remove_event_handler(lambda e: None)

    @pytest.mark.asyncio
    async def test_event_handler_exception_handled(self, router):
        def bad_handler(entry):
            raise ValueError("handler error")

        router.add_event_handler(bad_handler)
        # Should not raise
        decision = await router.route({"action": "test"})
        assert decision is not None

    @pytest.mark.asyncio
    async def test_multiple_event_handlers(self, router):
        calls_a: list = []
        calls_b: list = []
        router.add_event_handler(lambda e: calls_a.append(e))
        router.add_event_handler(lambda e: calls_b.append(e))
        await router.route({})
        assert len(calls_a) == 1
        assert len(calls_b) == 1


# =============================================================================
# DecisionRouter Tests - Configuration Updates
# =============================================================================


class TestConfigurationUpdates:
    """Tests for runtime configuration updates."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    def test_update_criteria(self, router):
        new_criteria = RoutingCriteria(financial_threshold=99999)
        router.update_criteria(new_criteria)
        assert router._criteria.financial_threshold == 99999

    def test_update_category_config(self, router):
        new_config = CategoryDefaults(
            category=ActionCategory.GENERAL,
            default_destination=RouteDestination.DEBATE,
        )
        router.update_category_config(ActionCategory.GENERAL, new_config)
        assert (
            router._category_configs[ActionCategory.GENERAL].default_destination
            == RouteDestination.DEBATE
        )

    def test_set_default_destination(self, router):
        router.set_default_destination(RouteDestination.REJECT)
        assert router._default_destination == RouteDestination.REJECT


# =============================================================================
# DecisionRouter Tests - Statistics
# =============================================================================


class TestStatistics:
    """Tests for router statistics."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_stats_structure(self, router):
        stats = await router.get_stats()
        assert "total_rules" in stats
        assert "enabled_rules" in stats
        assert "tenant_configs" in stats
        assert "category_configs" in stats
        assert "metrics" in stats
        assert "audit_entries" in stats
        assert "audit_enabled" in stats
        assert "default_destination" in stats
        assert "financial_threshold" in stats

    @pytest.mark.asyncio
    async def test_stats_reflect_rules(self, router):
        router.add_rule(
            RoutingRule(rule_id="r1", condition=lambda r: True, destination=RouteDestination.DEBATE)
        )
        router.add_rule(
            RoutingRule(
                rule_id="r2",
                condition=lambda r: True,
                destination=RouteDestination.EXECUTE,
                enabled=False,
            )
        )
        stats = await router.get_stats()
        assert stats["total_rules"] == 2
        assert stats["enabled_rules"] == 1

    @pytest.mark.asyncio
    async def test_stats_reflect_tenants(self, router):
        await router.add_tenant_config(TenantRoutingConfig(tenant_id="t1"))
        stats = await router.get_stats()
        assert stats["tenant_configs"] == 1

    @pytest.mark.asyncio
    async def test_stats_reflect_audit(self, router):
        await router.route({})
        stats = await router.get_stats()
        assert stats["audit_entries"] >= 1


# =============================================================================
# DecisionRouter Tests - Request ID Generation
# =============================================================================


class TestRequestIdGeneration:
    """Tests for request ID generation."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_request_id_generated(self, router):
        decision = await router.route({})
        assert decision.request_id.startswith("route-")
        assert len(decision.request_id) > 0

    @pytest.mark.asyncio
    async def test_request_ids_unique(self, router):
        d1 = await router.route({})
        d2 = await router.route({})
        assert d1.request_id != d2.request_id

    @pytest.mark.asyncio
    async def test_request_counter_increments(self, router):
        await router.route({})
        assert router._request_counter == 1
        await router.route({})
        assert router._request_counter == 2


# =============================================================================
# DecisionRouter Tests - Priority / Ordering
# =============================================================================


class TestRoutingPriority:
    """Tests for the priority ordering of routing checks."""

    @pytest.fixture
    def router(self):
        return DecisionRouter(
            criteria=RoutingCriteria(financial_threshold=10000),
        )

    @pytest.mark.asyncio
    async def test_explicit_intent_highest_priority(self, router):
        """Keywords should override financial threshold."""
        decision = await router.route(
            {
                "content": "Just execute this task",
                "amount": 50000,
            }
        )
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_custom_rules_before_financial(self, router):
        """Custom rules evaluated before financial threshold."""
        rule = RoutingRule(
            rule_id="override",
            condition=lambda req: req.get("override", False),
            destination=RouteDestination.EXECUTE,
            priority=100,
        )
        router.add_rule(rule)
        decision = await router.route({"amount": 50000, "override": True})
        assert decision.destination == RouteDestination.EXECUTE
        assert decision.rule_id == "override"

    @pytest.mark.asyncio
    async def test_financial_before_risk_level(self, router):
        """Financial threshold checked before risk level."""
        decision = await router.route({"amount": 50000, "risk_level": "high"})
        assert any("financial_threshold" in c for c in decision.criteria_matched)

    @pytest.mark.asyncio
    async def test_compliance_after_risk(self, router):
        """Risk level checked before compliance flags."""
        decision = await router.route(
            {
                "risk_level": "high",
                "compliance_flags": ["pii"],
            }
        )
        assert any("risk_level" in c for c in decision.criteria_matched)


# =============================================================================
# DecisionRouter Tests - Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    @pytest.mark.asyncio
    async def test_empty_request(self, router):
        decision = await router.route({})
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_none_context(self, router):
        decision = await router.route({}, context=None)
        assert decision is not None

    @pytest.mark.asyncio
    async def test_empty_context(self, router):
        decision = await router.route({}, context={})
        assert decision is not None

    @pytest.mark.asyncio
    async def test_decision_time_is_positive(self, router):
        decision = await router.route({})
        assert decision.decision_time_ms >= 0

    @pytest.mark.asyncio
    async def test_many_sequential_routes(self, router):
        for i in range(50):
            decision = await router.route({"action": f"task_{i}"})
            assert decision is not None
        assert router._metrics.total_requests == 50

    @pytest.mark.asyncio
    async def test_action_type_field(self, router):
        """action_type is alternative to action for category inference."""
        decision = await router.route({"action_type": "transfer_money"})
        assert decision.destination == RouteDestination.DEBATE  # inferred financial

    @pytest.mark.asyncio
    async def test_none_action_field(self, router):
        decision = await router.route({"action": None})
        assert decision is not None

    @pytest.mark.asyncio
    async def test_invalid_category_string(self, router):
        decision = await router.route({"category": "nonexistent_category"})
        # Invalid category -> falls through to action-based inference or GENERAL
        assert decision is not None

    @pytest.mark.asyncio
    async def test_amount_as_none(self, router):
        decision = await router.route({"amount": None})
        assert decision is not None


# =============================================================================
# SimpleAnomalyDetector Tests
# =============================================================================


class TestSimpleAnomalyDetector:
    """Tests for SimpleAnomalyDetector."""

    def test_defaults(self):
        detector = SimpleAnomalyDetector()
        assert detector.debate_ratio_high_threshold == 0.9
        assert detector.debate_ratio_low_threshold == 0.1
        assert detector.min_samples == 100

    def test_no_anomaly_below_min_samples(self):
        detector = SimpleAnomalyDetector(min_samples=100)
        metrics = RoutingMetrics(total_requests=50, debate_routes=50)
        is_anomaly, desc = detector.check_anomaly(None, RouteDestination.DEBATE, metrics)
        assert is_anomaly is False
        assert desc == ""

    def test_high_debate_ratio_anomaly(self):
        detector = SimpleAnomalyDetector(
            debate_ratio_high_threshold=0.8,
            min_samples=10,
        )
        metrics = RoutingMetrics(total_requests=100, debate_routes=90)
        is_anomaly, desc = detector.check_anomaly(None, RouteDestination.DEBATE, metrics)
        assert is_anomaly is True
        assert "high" in desc.lower()

    def test_low_debate_ratio_anomaly(self):
        detector = SimpleAnomalyDetector(
            debate_ratio_low_threshold=0.2,
            min_samples=10,
        )
        metrics = RoutingMetrics(total_requests=100, debate_routes=5, execute_routes=95)
        is_anomaly, desc = detector.check_anomaly(None, RouteDestination.EXECUTE, metrics)
        assert is_anomaly is True
        assert "low" in desc.lower()

    def test_normal_ratio_no_anomaly(self):
        detector = SimpleAnomalyDetector(
            debate_ratio_high_threshold=0.8,
            debate_ratio_low_threshold=0.2,
            min_samples=10,
        )
        metrics = RoutingMetrics(total_requests=100, debate_routes=50, execute_routes=50)
        is_anomaly, desc = detector.check_anomaly(None, RouteDestination.DEBATE, metrics)
        assert is_anomaly is False

    def test_tenant_id_passed(self):
        detector = SimpleAnomalyDetector(min_samples=1)
        metrics = RoutingMetrics(total_requests=10, debate_routes=5)
        # Should not crash with tenant_id
        is_anomaly, _ = detector.check_anomaly("acme", RouteDestination.DEBATE, metrics)
        assert isinstance(is_anomaly, bool)

    def test_exactly_at_high_threshold(self):
        detector = SimpleAnomalyDetector(
            debate_ratio_high_threshold=0.8,
            min_samples=1,
        )
        metrics = RoutingMetrics(total_requests=10, debate_routes=8)
        is_anomaly, _ = detector.check_anomaly(None, RouteDestination.DEBATE, metrics)
        # 8/10 = 0.8, which is not > 0.8
        assert is_anomaly is False

    def test_exactly_at_low_threshold(self):
        detector = SimpleAnomalyDetector(
            debate_ratio_low_threshold=0.1,
            min_samples=1,
        )
        metrics = RoutingMetrics(total_requests=10, debate_routes=1)
        is_anomaly, _ = detector.check_anomaly(None, RouteDestination.EXECUTE, metrics)
        # 1/10 = 0.1, which is not < 0.1
        assert is_anomaly is False


# =============================================================================
# DecisionRouter Tests - Concurrent Routing
# =============================================================================


class TestConcurrentRouting:
    """Tests for concurrent routing behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_routes(self):
        router = DecisionRouter()
        tasks = [router.route({"action": f"task_{i}"}) for i in range(20)]
        decisions = await asyncio.gather(*tasks)
        assert len(decisions) == 20
        assert all(d.destination is not None for d in decisions)

    @pytest.mark.asyncio
    async def test_concurrent_metrics_consistency(self):
        router = DecisionRouter()
        tasks = [router.route({}) for _ in range(50)]
        await asyncio.gather(*tasks)
        metrics = router.get_metrics()
        assert metrics["total_requests"] == 50
        total_destinations = (
            metrics["debate_routes"]
            + metrics["execute_routes"]
            + metrics["hybrid_routes"]
            + metrics["rejected_routes"]
        )
        assert total_destinations == 50


# =============================================================================
# DecisionRouter Tests - Action Category Inference
# =============================================================================


class TestActionCategoryInference:
    """Tests for _extract_action_category method."""

    @pytest.fixture
    def router(self):
        return DecisionRouter()

    def test_payment_inferred_as_financial(self, router):
        cat = router._extract_action_category({"action": "process_payment"})
        assert cat == ActionCategory.FINANCIAL

    def test_invoice_inferred_as_financial(self, router):
        cat = router._extract_action_category({"action": "send_invoice"})
        assert cat == ActionCategory.FINANCIAL

    def test_budget_inferred_as_financial(self, router):
        cat = router._extract_action_category({"action": "review_budget"})
        assert cat == ActionCategory.FINANCIAL

    def test_expense_inferred_as_financial(self, router):
        cat = router._extract_action_category({"action": "submit_expense"})
        assert cat == ActionCategory.FINANCIAL

    def test_audit_inferred_as_compliance(self, router):
        cat = router._extract_action_category({"action": "run_audit"})
        assert cat == ActionCategory.COMPLIANCE

    def test_policy_inferred_as_compliance(self, router):
        cat = router._extract_action_category({"action": "update_policy"})
        assert cat == ActionCategory.COMPLIANCE

    def test_regulation_inferred_as_compliance(self, router):
        cat = router._extract_action_category({"action": "check_regulation"})
        assert cat == ActionCategory.COMPLIANCE

    def test_access_inferred_as_security(self, router):
        cat = router._extract_action_category({"action": "manage_access"})
        assert cat == ActionCategory.SECURITY

    def test_permission_inferred_as_security(self, router):
        cat = router._extract_action_category({"action": "grant_permission"})
        assert cat == ActionCategory.SECURITY

    def test_auth_inferred_as_security(self, router):
        cat = router._extract_action_category({"action": "auth_token_refresh"})
        assert cat == ActionCategory.SECURITY

    def test_deploy_inferred_as_infrastructure(self, router):
        cat = router._extract_action_category({"action": "deploy_app"})
        assert cat == ActionCategory.INFRASTRUCTURE

    def test_server_inferred_as_infrastructure(self, router):
        cat = router._extract_action_category({"action": "restart_server"})
        assert cat == ActionCategory.INFRASTRUCTURE

    def test_database_inferred_as_infrastructure(self, router):
        cat = router._extract_action_category({"action": "scale_database"})
        assert cat == ActionCategory.INFRASTRUCTURE

    def test_backup_inferred_as_data_management(self, router):
        cat = router._extract_action_category({"action": "create_backup"})
        assert cat == ActionCategory.DATA_MANAGEMENT

    def test_migrate_inferred_as_data_management(self, router):
        cat = router._extract_action_category({"action": "migrate_schema"})
        assert cat == ActionCategory.DATA_MANAGEMENT

    def test_export_inferred_as_data_management(self, router):
        cat = router._extract_action_category({"action": "export_records"})
        assert cat == ActionCategory.DATA_MANAGEMENT

    def test_user_inferred_as_user_management(self, router):
        cat = router._extract_action_category({"action": "create_user"})
        assert cat == ActionCategory.USER_MANAGEMENT

    def test_account_inferred_as_user_management(self, router):
        cat = router._extract_action_category({"action": "deactivate_account"})
        assert cat == ActionCategory.USER_MANAGEMENT

    def test_role_inferred_as_user_management(self, router):
        cat = router._extract_action_category({"action": "assign_role"})
        assert cat == ActionCategory.USER_MANAGEMENT

    def test_unknown_action_defaults_to_general(self, router):
        cat = router._extract_action_category({"action": "random_stuff"})
        assert cat == ActionCategory.GENERAL

    def test_explicit_category_string(self, router):
        cat = router._extract_action_category({"category": "security"})
        assert cat == ActionCategory.SECURITY

    def test_explicit_action_category_field(self, router):
        cat = router._extract_action_category({"action_category": "analytics"})
        assert cat == ActionCategory.ANALYTICS

    def test_explicit_enum_value(self, router):
        cat = router._extract_action_category({"category": ActionCategory.COMPLIANCE})
        assert cat == ActionCategory.COMPLIANCE

    def test_no_action_at_all(self, router):
        cat = router._extract_action_category({})
        assert cat == ActionCategory.GENERAL


# =============================================================================
# DecisionRouter Tests - Average Decision Time
# =============================================================================


class TestAverageDecisionTime:
    """Tests for the running average decision time calculation."""

    def test_first_request_sets_time(self):
        router = DecisionRouter()
        router._update_avg_decision_time(router._metrics, 5.0)
        # total_requests not yet set here; relies on total_requests value
        # The method checks total <= 1
        assert router._metrics.avg_decision_time_ms == 5.0

    def test_running_average(self):
        m = RoutingMetrics(total_requests=2, avg_decision_time_ms=4.0)
        router = DecisionRouter()
        router._update_avg_decision_time(m, 6.0)
        # avg = (4.0 * 1 + 6.0) / 2 = 5.0
        assert abs(m.avg_decision_time_ms - 5.0) < 0.01


# =============================================================================
# Integration-Style Tests
# =============================================================================


class TestIntegration:
    """Integration-style tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test a complete routing lifecycle with rules, tenants, and metrics."""
        router = DecisionRouter(
            criteria=RoutingCriteria(financial_threshold=10000),
        )

        # Add a custom rule
        router.add_rule(
            RoutingRule(
                rule_id="emergency",
                condition=lambda req: req.get("emergency", False),
                destination=RouteDestination.REJECT,
                priority=1000,
                reason="Emergency halt",
            )
        )

        # Add a tenant config
        await router.add_tenant_config(
            TenantRoutingConfig(
                tenant_id="acme",
                criteria=RoutingCriteria(financial_threshold=5000),
            )
        )

        # Track events
        events: list = []
        router.add_event_handler(lambda e: events.append(e))

        # Route several requests
        d1 = await router.route({"action": "generic_task"})
        assert d1.destination == RouteDestination.EXECUTE

        d2 = await router.route({"amount": 50000})
        assert d2.destination == RouteDestination.DEBATE

        d3 = await router.route({"emergency": True})
        assert d3.destination == RouteDestination.REJECT

        d4 = await router.route(
            {"amount": 7000},
            context={"tenant_id": "acme"},
        )
        assert d4.destination == RouteDestination.DEBATE  # 7000 > tenant threshold 5000

        # Verify metrics
        metrics = router.get_metrics()
        assert metrics["total_requests"] == 4
        assert metrics["debate_routes"] == 2
        assert metrics["execute_routes"] == 1
        assert metrics["rejected_routes"] == 1

        # Verify tenant metrics
        tenant_metrics = router.get_tenant_metrics("acme")
        assert tenant_metrics is not None
        assert tenant_metrics["total_requests"] == 1

        # Verify audit log
        audit = await router.get_audit_log()
        assert len(audit) == 4

        # Verify events
        assert len(events) == 4

        # Verify stats
        stats = await router.get_stats()
        assert stats["total_rules"] == 1
        assert stats["tenant_configs"] == 1

    @pytest.mark.asyncio
    async def test_criteria_update_affects_routing(self):
        router = DecisionRouter(
            criteria=RoutingCriteria(financial_threshold=10000),
        )
        # Below threshold
        d1 = await router.route({"amount": 5000})
        assert d1.destination == RouteDestination.EXECUTE  # falls through

        # Update threshold
        router.update_criteria(RoutingCriteria(financial_threshold=1000))

        # Same amount now triggers debate
        d2 = await router.route({"amount": 5000})
        assert d2.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_disable_and_enable_rule(self):
        router = DecisionRouter()
        rule = RoutingRule(
            rule_id="toggle",
            condition=lambda req: True,
            destination=RouteDestination.REJECT,
            priority=100,
        )
        router.add_rule(rule)

        d1 = await router.route({})
        assert d1.destination == RouteDestination.REJECT

        router.disable_rule("toggle")
        d2 = await router.route({})
        assert d2.destination != RouteDestination.REJECT

        router.enable_rule("toggle")
        d3 = await router.route({})
        assert d3.destination == RouteDestination.REJECT
