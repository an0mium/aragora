"""Tests for external agent policy engine."""

import pytest

from aragora.gateway.external_agents.base import AgentCapability
from aragora.gateway.external_agents.policy import (
    PolicyEngine,
    PolicyDecision,
    PolicyAction,
    CapabilityRule,
    SensitivityLevel,
)


class TestPolicyAction:
    """Tests for PolicyAction enum."""

    def test_action_values(self):
        """Test policy action values."""
        assert PolicyAction.ALLOW.value == "allow"
        assert PolicyAction.DENY.value == "deny"
        assert PolicyAction.REQUIRE_APPROVAL.value == "require_approval"
        assert PolicyAction.AUDIT_ONLY.value == "audit_only"


class TestSensitivityLevel:
    """Tests for SensitivityLevel enum."""

    def test_sensitivity_levels(self):
        """Test sensitivity level values."""
        assert SensitivityLevel.PUBLIC.value == "public"
        assert SensitivityLevel.INTERNAL.value == "internal"
        assert SensitivityLevel.CONFIDENTIAL.value == "confidential"
        assert SensitivityLevel.RESTRICTED.value == "restricted"


class TestCapabilityRule:
    """Tests for CapabilityRule dataclass."""

    def test_default_rule(self):
        """Test rule with defaults."""
        rule = CapabilityRule(
            capability=AgentCapability.WEB_SEARCH,
            action=PolicyAction.ALLOW,
        )
        assert rule.capability == AgentCapability.WEB_SEARCH
        assert rule.action == PolicyAction.ALLOW
        assert rule.reason is None
        assert rule.tenant_ids is None

    def test_rule_with_tenant_restriction(self):
        """Test rule with tenant restriction."""
        rule = CapabilityRule(
            capability=AgentCapability.SHELL_ACCESS,
            action=PolicyAction.DENY,
            reason="Security policy",
            tenant_ids=["tenant-1", "tenant-2"],
        )
        assert rule.tenant_ids == ["tenant-1", "tenant-2"]
        assert rule.reason == "Security policy"


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""

    def test_allowed_decision(self):
        """Test allowed policy decision."""
        decision = PolicyDecision(
            allowed=True,
            action=PolicyAction.ALLOW,
        )
        assert decision.allowed is True
        assert decision.action == PolicyAction.ALLOW
        assert decision.requires_approval is False

    def test_denied_decision(self):
        """Test denied policy decision."""
        decision = PolicyDecision(
            allowed=False,
            action=PolicyAction.DENY,
            reason="Capability blocked",
        )
        assert decision.allowed is False
        assert decision.reason == "Capability blocked"

    def test_approval_required_decision(self):
        """Test approval required decision."""
        decision = PolicyDecision(
            allowed=False,
            action=PolicyAction.REQUIRE_APPROVAL,
            requires_approval=True,
            approval_gate="admin-approval",
        )
        assert decision.requires_approval is True
        assert decision.approval_gate == "admin-approval"


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    def test_default_engine(self):
        """Test engine with default rules."""
        engine = PolicyEngine()
        assert len(engine._rules) > 0  # Should have default rules

    def test_add_rule(self):
        """Test adding a rule."""
        engine = PolicyEngine(default_rules=[])
        rule = CapabilityRule(
            capability=AgentCapability.WEB_SEARCH,
            action=PolicyAction.ALLOW,
        )
        engine.add_rule(rule)
        assert AgentCapability.WEB_SEARCH in engine._rules

    def test_remove_rule(self):
        """Test removing a rule."""
        engine = PolicyEngine(default_rules=[])
        rule = CapabilityRule(
            capability=AgentCapability.WEB_SEARCH,
            action=PolicyAction.ALLOW,
        )
        engine.add_rule(rule)
        result = engine.remove_rule(AgentCapability.WEB_SEARCH)
        assert result is True
        assert AgentCapability.WEB_SEARCH not in engine._rules

    def test_check_allowed_capability(self):
        """Test checking allowed capability."""
        engine = PolicyEngine(default_rules=[])
        engine.add_rule(
            CapabilityRule(
                capability=AgentCapability.WEB_SEARCH,
                action=PolicyAction.ALLOW,
            )
        )

        decision = engine.check_capability(AgentCapability.WEB_SEARCH)
        assert decision.allowed is True
        assert decision.action == PolicyAction.ALLOW

    def test_check_denied_capability(self):
        """Test checking denied capability."""
        engine = PolicyEngine(default_rules=[])
        engine.add_rule(
            CapabilityRule(
                capability=AgentCapability.SHELL_ACCESS,
                action=PolicyAction.DENY,
                reason="Security policy",
            )
        )

        decision = engine.check_capability(AgentCapability.SHELL_ACCESS)
        assert decision.allowed is False
        assert decision.action == PolicyAction.DENY

    def test_check_unknown_capability_denied(self):
        """Test that unknown capabilities are denied by default."""
        engine = PolicyEngine(default_rules=[])
        # Don't add any rules
        decision = engine.check_capability(AgentCapability.WEB_SEARCH)
        assert decision.allowed is False

    def test_tenant_override(self):
        """Test tenant-specific override."""
        engine = PolicyEngine(default_rules=[])
        # Default deny
        engine.add_rule(
            CapabilityRule(
                capability=AgentCapability.SHELL_ACCESS,
                action=PolicyAction.DENY,
            )
        )
        # Allow for specific tenant
        engine.add_tenant_override(
            "tenant-123",
            AgentCapability.SHELL_ACCESS,
            PolicyAction.ALLOW,
        )

        # Without tenant - denied
        decision = engine.check_capability(AgentCapability.SHELL_ACCESS)
        assert decision.allowed is False

        # With tenant - allowed
        decision = engine.check_capability(
            AgentCapability.SHELL_ACCESS,
            tenant_id="tenant-123",
        )
        assert decision.allowed is True

    def test_check_multiple_capabilities(self):
        """Test checking multiple capabilities."""
        engine = PolicyEngine(default_rules=[])
        engine.add_rule(
            CapabilityRule(
                capability=AgentCapability.WEB_SEARCH,
                action=PolicyAction.ALLOW,
            )
        )
        engine.add_rule(
            CapabilityRule(
                capability=AgentCapability.EXECUTE_CODE,
                action=PolicyAction.ALLOW,
            )
        )
        engine.add_rule(
            CapabilityRule(
                capability=AgentCapability.SHELL_ACCESS,
                action=PolicyAction.DENY,
            )
        )

        decisions = engine.check_capabilities(
            [
                AgentCapability.WEB_SEARCH,
                AgentCapability.EXECUTE_CODE,
                AgentCapability.SHELL_ACCESS,
            ]
        )

        assert len(decisions) == 3
        assert decisions[AgentCapability.WEB_SEARCH].allowed is True
        assert decisions[AgentCapability.EXECUTE_CODE].allowed is True
        assert decisions[AgentCapability.SHELL_ACCESS].allowed is False

    def test_require_approval_action(self):
        """Test require approval action."""
        engine = PolicyEngine(default_rules=[])
        engine.add_rule(
            CapabilityRule(
                capability=AgentCapability.DATABASE_WRITE,
                action=PolicyAction.REQUIRE_APPROVAL,
                approval_gate="db-admin",
            )
        )

        decision = engine.check_capability(AgentCapability.DATABASE_WRITE)
        assert decision.allowed is False
        assert decision.requires_approval is True
        assert decision.approval_gate == "db-admin"

    def test_audit_only_action(self):
        """Test audit only action (allowed but logged)."""
        engine = PolicyEngine(default_rules=[])
        engine.add_rule(
            CapabilityRule(
                capability=AgentCapability.WEB_BROWSE,
                action=PolicyAction.AUDIT_ONLY,
            )
        )

        decision = engine.check_capability(AgentCapability.WEB_BROWSE)
        assert decision.allowed is True
        assert decision.action == PolicyAction.AUDIT_ONLY
