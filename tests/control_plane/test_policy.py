"""
Tests for Control Plane Policy Engine.

Tests cover:
- Policy creation and matching
- Agent restrictions (allowlist/blocklist)
- Region constraints and data residency
- SLA requirements and compliance
- Policy manager evaluation
- Violation tracking
- Factory functions
"""

import pytest
from datetime import datetime, timezone

from aragora.control_plane.policy import (
    ControlPlanePolicy,
    ControlPlanePolicyManager,
    EnforcementLevel,
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyScope,
    PolicyViolation,
    RegionConstraint,
    SLARequirements,
    create_agent_tier_policy,
    create_production_policy,
    create_sensitive_data_policy,
    create_sla_policy,
)


class TestSLARequirements:
    """Tests for SLARequirements."""

    def test_default_values(self):
        """Test default SLA values."""
        sla = SLARequirements()

        assert sla.max_execution_seconds == 300.0
        assert sla.max_queue_seconds == 60.0
        assert sla.min_agents_available == 1
        assert sla.max_concurrent_tasks == 5
        assert sla.response_time_p99_ms == 5000.0

    def test_execution_time_compliant(self):
        """Test execution time compliance check."""
        sla = SLARequirements(max_execution_seconds=60.0)

        assert sla.is_execution_time_compliant(30.0) is True
        assert sla.is_execution_time_compliant(60.0) is True
        assert sla.is_execution_time_compliant(90.0) is False

    def test_queue_time_compliant(self):
        """Test queue time compliance check."""
        sla = SLARequirements(max_queue_seconds=30.0)

        assert sla.is_queue_time_compliant(15.0) is True
        assert sla.is_queue_time_compliant(30.0) is True
        assert sla.is_queue_time_compliant(45.0) is False

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        original = SLARequirements(
            max_execution_seconds=120.0,
            max_queue_seconds=30.0,
            min_agents_available=3,
        )

        data = original.to_dict()
        restored = SLARequirements.from_dict(data)

        assert restored.max_execution_seconds == 120.0
        assert restored.max_queue_seconds == 30.0
        assert restored.min_agents_available == 3


class TestRegionConstraint:
    """Tests for RegionConstraint."""

    def test_default_allows_all(self):
        """Test that default constraint allows all regions."""
        constraint = RegionConstraint()

        assert constraint.is_region_allowed("us-east-1") is True
        assert constraint.is_region_allowed("eu-west-1") is True

    def test_explicit_allowlist(self):
        """Test explicit region allowlist."""
        constraint = RegionConstraint(allowed_regions=["us-east-1", "us-west-2"])

        assert constraint.is_region_allowed("us-east-1") is True
        assert constraint.is_region_allowed("us-west-2") is True
        assert constraint.is_region_allowed("eu-west-1") is False

    def test_explicit_blocklist(self):
        """Test explicit region blocklist."""
        constraint = RegionConstraint(blocked_regions=["ap-northeast-1"])

        assert constraint.is_region_allowed("us-east-1") is True
        assert constraint.is_region_allowed("ap-northeast-1") is False

    def test_blocklist_overrides_allowlist(self):
        """Test that blocklist takes precedence over allowlist."""
        constraint = RegionConstraint(
            allowed_regions=["us-east-1", "us-west-2"],
            blocked_regions=["us-east-1"],
        )

        assert constraint.is_region_allowed("us-east-1") is False
        assert constraint.is_region_allowed("us-west-2") is True

    def test_data_residency_requirement(self):
        """Test data residency enforcement."""
        constraint = RegionConstraint(require_data_residency=True)

        # Without data region specified, all allowed
        assert constraint.is_region_allowed("us-east-1") is True

        # With data region, must match
        assert constraint.is_region_allowed("us-east-1", data_region="us-east-1") is True
        assert constraint.is_region_allowed("eu-west-1", data_region="us-east-1") is False

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        original = RegionConstraint(
            allowed_regions=["us-east-1"],
            blocked_regions=["ap-northeast-1"],
            require_data_residency=True,
        )

        data = original.to_dict()
        restored = RegionConstraint.from_dict(data)

        assert restored.allowed_regions == ["us-east-1"]
        assert restored.blocked_regions == ["ap-northeast-1"]
        assert restored.require_data_residency is True


class TestControlPlanePolicy:
    """Tests for ControlPlanePolicy."""

    def test_default_policy_matches_all(self):
        """Test that a default policy matches everything."""
        policy = ControlPlanePolicy(name="allow-all")

        assert policy.matches(task_type="any-task") is True
        assert policy.matches(capability="any-cap") is True
        assert policy.matches(workspace="any-workspace") is True

    def test_policy_task_type_filter(self):
        """Test filtering by task type."""
        policy = ControlPlanePolicy(
            name="production-only",
            task_types=["production-deployment"],
        )

        assert policy.matches(task_type="production-deployment") is True
        assert policy.matches(task_type="dev-task") is False

    def test_policy_capability_filter(self):
        """Test filtering by capability."""
        policy = ControlPlanePolicy(
            name="code-only",
            capabilities=["code", "code-review"],
        )

        assert policy.matches(capability="code") is True
        assert policy.matches(capability="debate") is False

    def test_policy_workspace_filter(self):
        """Test filtering by workspace."""
        policy = ControlPlanePolicy(
            name="workspace-restricted",
            workspaces=["prod-workspace"],
        )

        assert policy.matches(workspace="prod-workspace") is True
        assert policy.matches(workspace="dev-workspace") is False

    def test_disabled_policy_never_matches(self):
        """Test that disabled policies don't match."""
        policy = ControlPlanePolicy(name="disabled", enabled=False)

        assert policy.matches() is False

    def test_agent_allowlist(self):
        """Test agent allowlist."""
        policy = ControlPlanePolicy(
            name="premium-agents",
            agent_allowlist=["claude-3-opus", "gpt-4"],
        )

        assert policy.is_agent_allowed("claude-3-opus") is True
        assert policy.is_agent_allowed("gpt-4") is True
        assert policy.is_agent_allowed("gpt-3.5-turbo") is False

    def test_agent_blocklist(self):
        """Test agent blocklist."""
        policy = ControlPlanePolicy(
            name="block-deprecated",
            agent_blocklist=["gpt-3.5-turbo", "claude-instant"],
        )

        assert policy.is_agent_allowed("claude-3-opus") is True
        assert policy.is_agent_allowed("gpt-3.5-turbo") is False

    def test_agent_blocklist_overrides_allowlist(self):
        """Test blocklist takes precedence."""
        policy = ControlPlanePolicy(
            name="mixed",
            agent_allowlist=["claude-3-opus", "gpt-4"],
            agent_blocklist=["gpt-4"],
        )

        assert policy.is_agent_allowed("claude-3-opus") is True
        assert policy.is_agent_allowed("gpt-4") is False

    def test_region_constraint_integration(self):
        """Test region constraint within policy."""
        policy = ControlPlanePolicy(
            name="us-only",
            region_constraint=RegionConstraint(
                allowed_regions=["us-east-1", "us-west-2"],
            ),
        )

        assert policy.is_region_allowed("us-east-1") is True
        assert policy.is_region_allowed("eu-west-1") is False

    def test_policy_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        original = ControlPlanePolicy(
            name="test-policy",
            description="Test description",
            scope=PolicyScope.TASK_TYPE,
            task_types=["task-a", "task-b"],
            agent_allowlist=["agent-1"],
            region_constraint=RegionConstraint(allowed_regions=["us-east-1"]),
            sla=SLARequirements(max_execution_seconds=60.0),
            enforcement_level=EnforcementLevel.SOFT,
            priority=50,
        )

        data = original.to_dict()
        restored = ControlPlanePolicy.from_dict(data)

        assert restored.name == "test-policy"
        assert restored.description == "Test description"
        assert restored.scope == PolicyScope.TASK_TYPE
        assert restored.task_types == ["task-a", "task-b"]
        assert restored.agent_allowlist == ["agent-1"]
        assert restored.region_constraint.allowed_regions == ["us-east-1"]
        assert restored.sla.max_execution_seconds == 60.0
        assert restored.enforcement_level == EnforcementLevel.SOFT
        assert restored.priority == 50


class TestControlPlanePolicyManager:
    """Tests for ControlPlanePolicyManager."""

    @pytest.fixture
    def manager(self):
        """Create a policy manager for testing."""
        return ControlPlanePolicyManager()

    def test_add_and_get_policy(self, manager):
        """Test adding and retrieving policies."""
        policy = ControlPlanePolicy(name="test")
        manager.add_policy(policy)

        retrieved = manager.get_policy(policy.id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_remove_policy(self, manager):
        """Test removing policies."""
        policy = ControlPlanePolicy(name="test")
        manager.add_policy(policy)

        result = manager.remove_policy(policy.id)
        assert result is True
        assert manager.get_policy(policy.id) is None

        # Removing non-existent policy returns False
        assert manager.remove_policy("nonexistent") is False

    def test_list_policies_sorted_by_priority(self, manager):
        """Test that policies are listed by priority."""
        policy1 = ControlPlanePolicy(name="low", priority=10)
        policy2 = ControlPlanePolicy(name="high", priority=100)
        policy3 = ControlPlanePolicy(name="medium", priority=50)

        manager.add_policy(policy1)
        manager.add_policy(policy2)
        manager.add_policy(policy3)

        policies = manager.list_policies()
        assert policies[0].name == "high"
        assert policies[1].name == "medium"
        assert policies[2].name == "low"

    def test_evaluate_allows_by_default(self, manager):
        """Test evaluation allows when no policies match."""
        result = manager.evaluate_task_dispatch(
            task_type="test-task",
            agent_id="claude-3",
            region="us-east-1",
        )

        assert result.decision == PolicyDecision.ALLOW
        assert result.allowed is True

    def test_evaluate_denies_blocked_agent(self, manager):
        """Test evaluation denies blocked agent."""
        policy = ControlPlanePolicy(
            name="block-agent",
            agent_blocklist=["blocked-agent"],
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="test-task",
            agent_id="blocked-agent",
            region="us-east-1",
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "blocked-agent" in result.reason

    def test_evaluate_denies_agent_not_in_allowlist(self, manager):
        """Test evaluation denies agent not in allowlist."""
        policy = ControlPlanePolicy(
            name="premium-only",
            agent_allowlist=["claude-3-opus", "gpt-4"],
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="test-task",
            agent_id="gpt-3.5-turbo",
            region="us-east-1",
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False

    def test_evaluate_denies_blocked_region(self, manager):
        """Test evaluation denies blocked region."""
        policy = ControlPlanePolicy(
            name="us-only",
            region_constraint=RegionConstraint(
                allowed_regions=["us-east-1", "us-west-2"],
            ),
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="test-task",
            agent_id="claude-3",
            region="eu-west-1",
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "eu-west-1" in result.reason

    def test_evaluate_warns_on_soft_enforcement(self, manager):
        """Test evaluation warns but allows with soft enforcement."""
        policy = ControlPlanePolicy(
            name="soft-policy",
            agent_blocklist=["warned-agent"],
            enforcement_level=EnforcementLevel.WARN,
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="test-task",
            agent_id="warned-agent",
            region="us-east-1",
        )

        assert result.decision == PolicyDecision.WARN
        assert result.allowed is True

    def test_evaluate_sla_compliance_passes(self, manager):
        """Test SLA compliance when within limits."""
        policy = ControlPlanePolicy(
            name="sla-policy",
            sla=SLARequirements(
                max_execution_seconds=60.0,
                max_queue_seconds=30.0,
            ),
        )
        manager.add_policy(policy)

        result = manager.evaluate_sla_compliance(
            policy_id=policy.id,
            execution_seconds=45.0,
            queue_seconds=20.0,
        )

        assert result.decision == PolicyDecision.ALLOW
        assert result.allowed is True

    def test_evaluate_sla_compliance_fails_execution_time(self, manager):
        """Test SLA compliance failure on execution time."""
        policy = ControlPlanePolicy(
            name="sla-policy",
            sla=SLARequirements(max_execution_seconds=60.0),
        )
        manager.add_policy(policy)

        result = manager.evaluate_sla_compliance(
            policy_id=policy.id,
            execution_seconds=90.0,
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "Execution time" in result.sla_violation

    def test_evaluate_sla_compliance_fails_queue_time(self, manager):
        """Test SLA compliance failure on queue time."""
        policy = ControlPlanePolicy(
            name="sla-policy",
            sla=SLARequirements(max_queue_seconds=30.0),
        )
        manager.add_policy(policy)

        result = manager.evaluate_sla_compliance(
            policy_id=policy.id,
            queue_seconds=45.0,
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "Queue time" in result.sla_violation

    def test_evaluate_sla_compliance_fails_agents_available(self, manager):
        """Test SLA compliance failure on available agents."""
        policy = ControlPlanePolicy(
            name="sla-policy",
            sla=SLARequirements(min_agents_available=3),
        )
        manager.add_policy(policy)

        result = manager.evaluate_sla_compliance(
            policy_id=policy.id,
            available_agents=1,
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "Available agents" in result.sla_violation

    def test_violation_tracking(self, manager):
        """Test that violations are recorded."""
        policy = ControlPlanePolicy(
            name="strict-policy",
            agent_blocklist=["bad-agent"],
        )
        manager.add_policy(policy)

        # Trigger a violation
        manager.evaluate_task_dispatch(
            task_type="test-task",
            agent_id="bad-agent",
            region="us-east-1",
            task_id="task-123",
        )

        violations = manager.get_violations()
        assert len(violations) == 1
        assert violations[0].violation_type == "agent"
        assert violations[0].policy_id == policy.id

    def test_violation_callback(self):
        """Test violation callback is invoked."""
        violations_received = []

        def callback(violation):
            violations_received.append(violation)

        manager = ControlPlanePolicyManager(violation_callback=callback)
        policy = ControlPlanePolicy(
            name="callback-test",
            agent_blocklist=["trigger-agent"],
        )
        manager.add_policy(policy)

        manager.evaluate_task_dispatch(
            task_type="test-task",
            agent_id="trigger-agent",
            region="us-east-1",
        )

        assert len(violations_received) == 1
        assert violations_received[0].policy_name == "callback-test"

    def test_metrics_tracking(self, manager):
        """Test metrics are tracked correctly."""
        policy = ControlPlanePolicy(name="test-policy")
        manager.add_policy(policy)

        # Run some evaluations
        manager.evaluate_task_dispatch("task", "agent", "region")
        manager.evaluate_task_dispatch("task", "agent", "region")

        metrics = manager.get_metrics()
        assert metrics["evaluations"] == 2
        assert metrics["policy_count"] == 1

    def test_clear_violations(self, manager):
        """Test clearing violations."""
        policy = ControlPlanePolicy(
            name="test",
            agent_blocklist=["blocked"],
        )
        manager.add_policy(policy)

        # Generate violations
        manager.evaluate_task_dispatch("task", "blocked", "region")
        manager.evaluate_task_dispatch("task", "blocked", "region")

        assert len(manager.get_violations()) == 2

        cleared = manager.clear_violations()
        assert cleared == 2
        assert len(manager.get_violations()) == 0


class TestFactoryFunctions:
    """Tests for policy factory functions."""

    def test_create_production_policy(self):
        """Test production policy factory."""
        policy = create_production_policy(
            agent_allowlist=["claude-3-opus", "gpt-4"],
            allowed_regions=["us-east-1"],
        )

        assert policy.name == "production-restrictions"
        assert "production-deployment" in policy.task_types
        assert "claude-3-opus" in policy.agent_allowlist
        assert policy.region_constraint is not None
        assert "us-east-1" in policy.region_constraint.allowed_regions
        assert policy.sla is not None
        assert policy.enforcement_level == EnforcementLevel.HARD

    def test_create_sensitive_data_policy(self):
        """Test sensitive data policy factory."""
        policy = create_sensitive_data_policy(
            data_regions=["us-east-1", "us-west-2"],
            blocked_regions=["cn-north-1"],
        )

        assert policy.name == "sensitive-data-residency"
        assert "pii-processing" in policy.task_types
        assert policy.region_constraint.require_data_residency is True
        assert "cn-north-1" in policy.region_constraint.blocked_regions

    def test_create_agent_tier_policy(self):
        """Test agent tier policy factory."""
        policy = create_agent_tier_policy(
            tier="premium",
            agents=["claude-3-opus", "gpt-4o"],
            task_types=["high-value-task"],
        )

        assert policy.name == "premium-agent-tier"
        assert "claude-3-opus" in policy.agent_allowlist
        assert "high-value-task" in policy.task_types

    def test_create_sla_policy(self):
        """Test SLA policy factory."""
        policy = create_sla_policy(
            name="fast-response",
            task_types=["realtime-query"],
            max_execution_seconds=10.0,
            max_queue_seconds=5.0,
        )

        assert policy.name == "fast-response"
        assert policy.sla is not None
        assert policy.sla.max_execution_seconds == 10.0
        assert policy.sla.max_queue_seconds == 5.0
        assert policy.enforcement_level == EnforcementLevel.WARN


class TestPolicyViolation:
    """Tests for PolicyViolation dataclass."""

    def test_violation_creation(self):
        """Test creating a violation."""
        violation = PolicyViolation(
            id="violation-123",
            policy_id="policy-456",
            policy_name="test-policy",
            violation_type="agent",
            description="Agent not allowed",
            task_id="task-789",
            agent_id="bad-agent",
        )

        assert violation.id == "violation-123"
        assert violation.violation_type == "agent"
        assert violation.description == "Agent not allowed"

    def test_violation_serialization(self):
        """Test violation to_dict."""
        violation = PolicyViolation(
            id="violation-123",
            policy_id="policy-456",
            policy_name="test-policy",
            violation_type="region",
            description="Region blocked",
            region="eu-west-1",
        )

        data = violation.to_dict()

        assert data["id"] == "violation-123"
        assert data["violation_type"] == "region"
        assert data["region"] == "eu-west-1"


class TestPolicyEvaluationResult:
    """Tests for PolicyEvaluationResult dataclass."""

    def test_result_creation(self):
        """Test creating an evaluation result."""
        result = PolicyEvaluationResult(
            decision=PolicyDecision.DENY,
            allowed=False,
            policy_id="policy-123",
            policy_name="test-policy",
            reason="Agent blocked",
            enforcement_level=EnforcementLevel.HARD,
            agent_id="blocked-agent",
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert result.agent_id == "blocked-agent"

    def test_result_serialization(self):
        """Test result to_dict."""
        result = PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW,
            allowed=True,
            policy_id="",
            policy_name="",
            reason="All policies passed",
            enforcement_level=EnforcementLevel.HARD,
        )

        data = result.to_dict()

        assert data["decision"] == "allow"
        assert data["allowed"] is True


class TestEnforcementLevels:
    """Tests for different enforcement levels."""

    def test_warn_level_allows_violation(self):
        """Test WARN level allows but logs."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="warn-only",
            agent_blocklist=["test-agent"],
            enforcement_level=EnforcementLevel.WARN,
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="task",
            agent_id="test-agent",
            region="us-east-1",
        )

        assert result.decision == PolicyDecision.WARN
        assert result.allowed is True
        assert len(manager.get_violations()) == 1

    def test_soft_level_denies(self):
        """Test SOFT level denies but can be overridden."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="soft-deny",
            agent_blocklist=["test-agent"],
            enforcement_level=EnforcementLevel.SOFT,
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="task",
            agent_id="test-agent",
            region="us-east-1",
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert result.enforcement_level == EnforcementLevel.SOFT

    def test_hard_level_strictly_denies(self):
        """Test HARD level strictly denies."""
        manager = ControlPlanePolicyManager()
        policy = ControlPlanePolicy(
            name="hard-deny",
            agent_blocklist=["test-agent"],
            enforcement_level=EnforcementLevel.HARD,
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="task",
            agent_id="test-agent",
            region="us-east-1",
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert result.enforcement_level == EnforcementLevel.HARD
