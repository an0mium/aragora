"""Tests for control plane policy manager."""

from __future__ import annotations

import pytest

from aragora.control_plane.policy.manager import ControlPlanePolicyManager
from aragora.control_plane.policy.types import (
    ControlPlanePolicy,
    EnforcementLevel,
    PolicyDecision,
    PolicyScope,
    PolicyViolation,
    RegionConstraint,
    SLARequirements,
)


class TestControlPlanePolicyManager:
    """Tests for ControlPlanePolicyManager."""

    def test_empty_manager(self):
        """Test empty manager allows everything."""
        manager = ControlPlanePolicyManager()
        result = manager.evaluate_task_dispatch(
            task_type="any-task",
            agent_id="any-agent",
            region="us-east-1",
        )
        assert result.allowed is True
        assert result.decision == PolicyDecision.ALLOW

    def test_add_and_get_policy(self):
        """Test adding and retrieving a policy."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="test-policy",
            description="A test policy",
        )
        manager.add_policy(policy)

        retrieved = manager.get_policy(policy.id)
        assert retrieved is not None
        assert retrieved.name == "test-policy"

    def test_remove_policy(self):
        """Test removing a policy."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(name="to-remove")
        manager.add_policy(policy)

        assert manager.remove_policy(policy.id) is True
        assert manager.get_policy(policy.id) is None
        assert manager.remove_policy(policy.id) is False  # Already removed

    def test_list_policies(self):
        """Test listing all policies."""
        manager = ControlPlanePolicyManager()
        policy1 = ControlPlanePolicy(name="policy-1")
        policy2 = ControlPlanePolicy(name="policy-2")

        manager.add_policy(policy1)
        manager.add_policy(policy2)

        policies = manager.list_policies()
        assert len(policies) == 2
        names = [p.name for p in policies]
        assert "policy-1" in names
        assert "policy-2" in names

    def test_agent_allowlist_enforcement(self):
        """Test agent allowlist is enforced."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="agent-restriction",
            agent_allowlist=["claude-3-opus", "gpt-4"],
            enforcement_level=EnforcementLevel.HARD,
        )
        manager.add_policy(policy)

        # Allowed agent
        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="claude-3-opus",
            region="us-east-1",
        )
        assert result.allowed is True

        # Blocked agent
        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="gpt-3.5-turbo",
            region="us-east-1",
        )
        assert result.allowed is False
        assert result.decision == PolicyDecision.DENY

    def test_agent_blocklist_enforcement(self):
        """Test agent blocklist is enforced."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="block-untrusted",
            agent_blocklist=["untrusted-agent", "deprecated-agent"],
        )
        manager.add_policy(policy)

        # Allowed agent
        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="trusted-agent",
            region="us-east-1",
        )
        assert result.allowed is True

        # Blocked agent
        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="untrusted-agent",
            region="us-east-1",
        )
        assert result.allowed is False

    def test_region_constraint_enforcement(self):
        """Test region constraints are enforced."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="region-restriction",
            region_constraint=RegionConstraint(
                allowed_regions=["us-east-1", "us-west-2"],
            ),
        )
        manager.add_policy(policy)

        # Allowed region
        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="any",
            region="us-east-1",
        )
        assert result.allowed is True

        # Blocked region
        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="any",
            region="eu-west-1",
        )
        assert result.allowed is False

    def test_task_type_scoped_policy(self):
        """Test policy only applies to specific task types."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="production-only",
            task_types=["production-deploy"],
            agent_allowlist=["claude-3-opus"],
        )
        manager.add_policy(policy)

        # Policy applies - agent blocked
        result = manager.evaluate_task_dispatch(
            task_type="production-deploy",
            agent_id="gpt-4",
            region="us-east-1",
        )
        assert result.allowed is False

        # Policy doesn't apply - agent allowed
        result = manager.evaluate_task_dispatch(
            task_type="development-test",
            agent_id="gpt-4",
            region="us-east-1",
        )
        assert result.allowed is True

    def test_enforcement_level_warn(self):
        """Test WARN enforcement level."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="warn-policy",
            agent_blocklist=["deprecated-agent"],
            enforcement_level=EnforcementLevel.WARN,
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="deprecated-agent",
            region="us-east-1",
        )
        # WARN allows but logs warning
        assert result.allowed is True
        assert result.decision == PolicyDecision.WARN

    def test_enforcement_level_soft(self):
        """Test SOFT enforcement level."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="soft-policy",
            agent_blocklist=["soft-blocked"],
            enforcement_level=EnforcementLevel.SOFT,
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="soft-blocked",
            region="us-east-1",
        )
        # SOFT denies but can be overridden
        assert result.allowed is False
        assert result.can_override is True

    def test_multiple_policies_most_restrictive(self):
        """Test multiple policies - most restrictive wins."""
        manager = ControlPlanePolicyManager()

        # Lenient policy
        policy1 = ControlPlanePolicy(
            name="lenient",
            enforcement_level=EnforcementLevel.WARN,
            agent_blocklist=["agent-x"],
            priority=1,
        )
        # Strict policy
        policy2 = ControlPlanePolicy(
            name="strict",
            enforcement_level=EnforcementLevel.HARD,
            agent_blocklist=["agent-x"],
            priority=2,  # Higher priority
        )

        manager.add_policy(policy1)
        manager.add_policy(policy2)

        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="agent-x",
            region="us-east-1",
        )
        # Strict policy should win
        assert result.allowed is False
        assert result.enforcement_level == EnforcementLevel.HARD

    def test_disabled_policy_ignored(self):
        """Test disabled policies are ignored."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="disabled",
            agent_blocklist=["blocked-agent"],
            enabled=False,
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="blocked-agent",
            region="us-east-1",
        )
        assert result.allowed is True

    def test_violation_callback(self):
        """Test violation callback is called."""
        violations = []

        def on_violation(violation: PolicyViolation):
            violations.append(violation)

        manager = ControlPlanePolicyManager(violation_callback=on_violation)
        policy = ControlPlanePolicy(
            name="strict",
            agent_blocklist=["blocked"],
        )
        manager.add_policy(policy)

        manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="blocked",
            region="us-east-1",
        )

        assert len(violations) == 1
        assert violations[0].policy_name == "strict"

    def test_metrics_tracking(self):
        """Test metrics are tracked."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="test",
            agent_blocklist=["blocked"],
        )
        manager.add_policy(policy)

        # Allowed request
        manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="allowed",
            region="us-east-1",
        )
        # Denied request
        manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="blocked",
            region="us-east-1",
        )

        metrics = manager.get_metrics()
        assert metrics["evaluations"] == 2
        assert metrics["allowed"] >= 1
        assert metrics["denied"] >= 1

    def test_get_violations(self):
        """Test getting violation history."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="strict",
            agent_blocklist=["blocked"],
        )
        manager.add_policy(policy)

        # Generate violations
        for _ in range(3):
            manager.evaluate_task_dispatch(
                task_type="any",
                agent_id="blocked",
                region="us-east-1",
            )

        violations = manager.get_violations()
        assert len(violations) == 3

    def test_clear_violations(self):
        """Test clearing violation history."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="strict",
            agent_blocklist=["blocked"],
        )
        manager.add_policy(policy)

        manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="blocked",
            region="us-east-1",
        )

        assert len(manager.get_violations()) == 1
        manager.clear_violations()
        assert len(manager.get_violations()) == 0

    def test_sla_requirements_check(self):
        """Test SLA requirements are included in evaluation."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="sla-policy",
            sla=SLARequirements(
                max_execution_seconds=60.0,
                max_queue_seconds=10.0,
            ),
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="any",
            region="us-east-1",
        )
        # SLA is informational, doesn't block
        assert result.allowed is True
        assert result.sla_requirements is not None
        assert result.sla_requirements.max_execution_seconds == 60.0


class TestPolicyPriority:
    """Tests for policy priority handling."""

    def test_higher_priority_evaluated_first(self):
        """Test higher priority policies are evaluated first."""
        manager = ControlPlanePolicyManager()

        # Low priority allows agent-x
        low_priority = ControlPlanePolicy(
            name="low",
            priority=1,
            agent_allowlist=["agent-x", "agent-y"],
        )
        # High priority blocks agent-x
        high_priority = ControlPlanePolicy(
            name="high",
            priority=10,
            agent_blocklist=["agent-x"],
        )

        manager.add_policy(low_priority)
        manager.add_policy(high_priority)

        result = manager.evaluate_task_dispatch(
            task_type="any",
            agent_id="agent-x",
            region="us-east-1",
        )
        # High priority blocklist should take effect
        assert result.allowed is False
        assert result.policy_name == "high"
