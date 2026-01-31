"""Tests for external agent policy."""

import pytest

from aragora.gateway.external_agents.base import AgentCapability
from aragora.gateway.external_agents.policy import (
    ExternalAgentPolicy,
    PolicyDecision,
    PolicyAction,
    PolicyViolation,
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
            capability="web_search",
            action=PolicyAction.ALLOW,
        )
        assert rule.capability == "web_search"
        assert rule.action == PolicyAction.ALLOW

    def test_rule_with_conditions(self):
        """Test rule with conditions."""
        rule = CapabilityRule(
            capability="shell_access",
            action=PolicyAction.DENY,
            conditions={"tenant_ids": ["tenant-1", "tenant-2"]},
        )
        assert rule.conditions["tenant_ids"] == ["tenant-1", "tenant-2"]


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
            approval_id="approval-123",
        )
        assert decision.requires_approval is True
        assert decision.approval_id == "approval-123"


class TestPolicyViolation:
    """Tests for PolicyViolation dataclass."""

    def test_violation_creation(self):
        """Test violation creation."""
        violation = PolicyViolation(
            policy_id="policy-123",
            policy_name="Security Policy",
            violation_type="capability_blocked",
            severity="high",
            details="Shell access is not allowed",
        )
        assert violation.policy_id == "policy-123"
        assert violation.severity == "high"


class TestExternalAgentPolicy:
    """Tests for ExternalAgentPolicy."""

    def test_default_policy(self):
        """Test policy with default values."""
        policy = ExternalAgentPolicy(
            policy_id="test-policy",
            policy_name="Test Policy",
        )
        assert policy.enabled is True
        assert policy.default_action == PolicyAction.DENY
        assert policy.max_executions_per_minute == 60

    def test_policy_with_capability_rules(self):
        """Test policy with capability rules."""
        policy = ExternalAgentPolicy(
            policy_id="test-policy",
            policy_name="Test Policy",
            capability_rules=[
                CapabilityRule(
                    capability="web_search",
                    action=PolicyAction.ALLOW,
                ),
                CapabilityRule(
                    capability="shell_access",
                    action=PolicyAction.DENY,
                ),
            ],
        )
        assert len(policy.capability_rules) == 2

    def test_get_capability_action_allow(self):
        """Test getting action for allowed capability."""
        policy = ExternalAgentPolicy(
            policy_id="test-policy",
            policy_name="Test Policy",
            capability_rules=[
                CapabilityRule(
                    capability="web_search",
                    action=PolicyAction.ALLOW,
                ),
            ],
        )
        action = policy.get_capability_action("web_search")
        assert action == PolicyAction.ALLOW

    def test_get_capability_action_deny(self):
        """Test getting action for denied capability."""
        policy = ExternalAgentPolicy(
            policy_id="test-policy",
            policy_name="Test Policy",
            capability_rules=[
                CapabilityRule(
                    capability="shell_access",
                    action=PolicyAction.DENY,
                ),
            ],
        )
        action = policy.get_capability_action("shell_access")
        assert action == PolicyAction.DENY

    def test_get_capability_action_default(self):
        """Test default action for unknown capability."""
        policy = ExternalAgentPolicy(
            policy_id="test-policy",
            policy_name="Test Policy",
            default_action=PolicyAction.DENY,
        )
        action = policy.get_capability_action("unknown_capability")
        assert action == PolicyAction.DENY

    def test_tenant_override(self):
        """Test tenant-specific override."""
        policy = ExternalAgentPolicy(
            policy_id="test-policy",
            policy_name="Test Policy",
            capability_rules=[
                CapabilityRule(
                    capability="shell_access",
                    action=PolicyAction.DENY,
                ),
            ],
            tenant_overrides={
                "tenant-123": {
                    "allowed_capabilities": ["shell_access"],
                }
            },
        )

        # Without tenant - denied
        action = policy.get_capability_action("shell_access")
        assert action == PolicyAction.DENY

        # With tenant override - allowed
        action = policy.get_capability_action("shell_access", tenant_id="tenant-123")
        assert action == PolicyAction.ALLOW

    def test_tenant_blocked_capabilities(self):
        """Test tenant-blocked capabilities."""
        policy = ExternalAgentPolicy(
            policy_id="test-policy",
            policy_name="Test Policy",
            capability_rules=[
                CapabilityRule(
                    capability="web_search",
                    action=PolicyAction.ALLOW,
                ),
            ],
            tenant_overrides={
                "tenant-456": {
                    "blocked_capabilities": ["web_search"],
                }
            },
        )

        # Default - allowed
        action = policy.get_capability_action("web_search")
        assert action == PolicyAction.ALLOW

        # With tenant block - denied
        action = policy.get_capability_action("web_search", tenant_id="tenant-456")
        assert action == PolicyAction.DENY

    def test_blocked_agents(self):
        """Test blocked agents list."""
        policy = ExternalAgentPolicy(
            policy_id="test-policy",
            policy_name="Test Policy",
            blocked_agents=["unsafe-agent", "untrusted-agent"],
        )
        assert "unsafe-agent" in policy.blocked_agents
        assert "untrusted-agent" in policy.blocked_agents

    def test_allowed_agents(self):
        """Test allowed agents list."""
        policy = ExternalAgentPolicy(
            policy_id="test-policy",
            policy_name="Test Policy",
            allowed_agents=["openclaw", "openhands"],
        )
        assert "openclaw" in policy.allowed_agents
        assert "openhands" in policy.allowed_agents
