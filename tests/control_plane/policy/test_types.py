"""Tests for control plane policy types."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aragora.control_plane.policy.types import (
    ControlPlanePolicy,
    EnforcementLevel,
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyScope,
    PolicyViolation,
    PolicyViolationError,
    RegionConstraint,
    SLARequirements,
)


class TestSLARequirements:
    """Tests for SLARequirements dataclass."""

    def test_default_values(self):
        """Test default SLA values."""
        sla = SLARequirements()
        assert sla.max_execution_seconds == 300.0
        assert sla.max_queue_seconds == 60.0
        assert sla.min_agents_available == 1
        assert sla.max_concurrent_tasks == 5
        assert sla.response_time_p99_ms == 5000.0

    def test_custom_values(self):
        """Test custom SLA values."""
        sla = SLARequirements(
            max_execution_seconds=600.0,
            max_queue_seconds=120.0,
            min_agents_available=3,
            max_concurrent_tasks=10,
            response_time_p99_ms=2000.0,
        )
        assert sla.max_execution_seconds == 600.0
        assert sla.max_queue_seconds == 120.0
        assert sla.min_agents_available == 3
        assert sla.max_concurrent_tasks == 10
        assert sla.response_time_p99_ms == 2000.0

    def test_is_execution_time_compliant(self):
        """Test execution time compliance check."""
        sla = SLARequirements(max_execution_seconds=100.0)
        assert sla.is_execution_time_compliant(50.0) is True
        assert sla.is_execution_time_compliant(100.0) is True
        assert sla.is_execution_time_compliant(101.0) is False

    def test_is_queue_time_compliant(self):
        """Test queue time compliance check."""
        sla = SLARequirements(max_queue_seconds=30.0)
        assert sla.is_queue_time_compliant(15.0) is True
        assert sla.is_queue_time_compliant(30.0) is True
        assert sla.is_queue_time_compliant(31.0) is False

    def test_to_dict(self):
        """Test serialization to dict."""
        sla = SLARequirements(
            max_execution_seconds=100.0,
            max_queue_seconds=30.0,
        )
        data = sla.to_dict()
        assert data["max_execution_seconds"] == 100.0
        assert data["max_queue_seconds"] == 30.0
        assert "min_agents_available" in data
        assert "max_concurrent_tasks" in data
        assert "response_time_p99_ms" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "max_execution_seconds": 150.0,
            "max_queue_seconds": 45.0,
            "min_agents_available": 2,
        }
        sla = SLARequirements.from_dict(data)
        assert sla.max_execution_seconds == 150.0
        assert sla.max_queue_seconds == 45.0
        assert sla.min_agents_available == 2
        # Defaults for missing values
        assert sla.max_concurrent_tasks == 5

    def test_from_dict_with_defaults(self):
        """Test deserialization with missing values uses defaults."""
        sla = SLARequirements.from_dict({})
        assert sla.max_execution_seconds == 300.0
        assert sla.max_queue_seconds == 60.0


class TestRegionConstraint:
    """Tests for RegionConstraint dataclass."""

    def test_default_allows_all(self):
        """Test default constraint allows all regions."""
        constraint = RegionConstraint()
        assert constraint.is_region_allowed("us-east-1") is True
        assert constraint.is_region_allowed("eu-west-1") is True
        assert constraint.is_region_allowed("ap-southeast-1") is True

    def test_allowed_regions(self):
        """Test explicit allowlist."""
        constraint = RegionConstraint(
            allowed_regions=["us-east-1", "us-west-2"],
        )
        assert constraint.is_region_allowed("us-east-1") is True
        assert constraint.is_region_allowed("us-west-2") is True
        assert constraint.is_region_allowed("eu-west-1") is False

    def test_blocked_regions(self):
        """Test explicit blocklist."""
        constraint = RegionConstraint(
            blocked_regions=["cn-north-1", "ru-central-1"],
        )
        assert constraint.is_region_allowed("us-east-1") is True
        assert constraint.is_region_allowed("cn-north-1") is False
        assert constraint.is_region_allowed("ru-central-1") is False

    def test_blocklist_takes_precedence(self):
        """Test blocklist overrides allowlist."""
        constraint = RegionConstraint(
            allowed_regions=["us-east-1", "us-west-2"],
            blocked_regions=["us-west-2"],
        )
        assert constraint.is_region_allowed("us-east-1") is True
        assert constraint.is_region_allowed("us-west-2") is False

    def test_data_residency_same_region(self):
        """Test data residency with same region."""
        constraint = RegionConstraint(require_data_residency=True)
        assert constraint.is_region_allowed("eu-west-1", data_region="eu-west-1") is True

    def test_data_residency_different_region(self):
        """Test data residency with different region."""
        constraint = RegionConstraint(require_data_residency=True)
        assert constraint.is_region_allowed("us-east-1", data_region="eu-west-1") is False

    def test_data_residency_no_data_region(self):
        """Test data residency when data_region not specified."""
        constraint = RegionConstraint(require_data_residency=True)
        # If data_region is None, residency check is skipped
        assert constraint.is_region_allowed("us-east-1", data_region=None) is True

    def test_to_dict(self):
        """Test serialization."""
        constraint = RegionConstraint(
            allowed_regions=["us-east-1"],
            blocked_regions=["cn-north-1"],
            require_data_residency=True,
            allow_cross_region=False,
        )
        data = constraint.to_dict()
        assert data["allowed_regions"] == ["us-east-1"]
        assert data["blocked_regions"] == ["cn-north-1"]
        assert data["require_data_residency"] is True
        assert data["allow_cross_region"] is False

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "allowed_regions": ["eu-west-1"],
            "require_data_residency": True,
        }
        constraint = RegionConstraint.from_dict(data)
        assert constraint.allowed_regions == ["eu-west-1"]
        assert constraint.require_data_residency is True
        assert constraint.blocked_regions == []


class TestControlPlanePolicy:
    """Tests for ControlPlanePolicy dataclass."""

    def test_default_policy(self):
        """Test policy with default values."""
        policy = ControlPlanePolicy(name="test-policy")
        assert policy.name == "test-policy"
        assert policy.enabled is True
        assert policy.scope == PolicyScope.GLOBAL
        assert policy.enforcement_level == EnforcementLevel.HARD
        assert policy.id.startswith("policy_")

    def test_policy_with_agent_restrictions(self):
        """Test policy with agent allowlist/blocklist."""
        policy = ControlPlanePolicy(
            name="agent-restricted",
            agent_allowlist=["claude-3-opus", "gpt-4"],
            agent_blocklist=["gpt-3.5-turbo"],
        )
        assert policy.is_agent_allowed("claude-3-opus") is True
        assert policy.is_agent_allowed("gpt-4") is True
        assert policy.is_agent_allowed("gpt-3.5-turbo") is False
        assert policy.is_agent_allowed("llama-2") is False  # Not in allowlist

    def test_policy_agent_blocklist_only(self):
        """Test policy with only blocklist."""
        policy = ControlPlanePolicy(
            name="blocklist-only",
            agent_blocklist=["untrusted-agent"],
        )
        assert policy.is_agent_allowed("claude-3-opus") is True
        assert policy.is_agent_allowed("untrusted-agent") is False

    def test_policy_matches_all_when_empty(self):
        """Test global policy matches everything."""
        policy = ControlPlanePolicy(name="global-policy")
        assert policy.matches() is True
        assert policy.matches(task_type="any-task") is True
        assert policy.matches(capability="any-cap") is True
        assert policy.matches(workspace="any-workspace") is True

    def test_policy_matches_specific_task_types(self):
        """Test policy with specific task types."""
        policy = ControlPlanePolicy(
            name="task-specific",
            task_types=["production-deploy", "staging-deploy"],
        )
        assert policy.matches(task_type="production-deploy") is True
        assert policy.matches(task_type="staging-deploy") is True
        assert policy.matches(task_type="development-test") is False

    def test_policy_matches_specific_capabilities(self):
        """Test policy with specific capabilities."""
        policy = ControlPlanePolicy(
            name="capability-specific",
            capabilities=["write", "delete"],
        )
        assert policy.matches(capability="write") is True
        assert policy.matches(capability="delete") is True
        assert policy.matches(capability="read") is False

    def test_policy_matches_specific_workspaces(self):
        """Test policy with specific workspaces."""
        policy = ControlPlanePolicy(
            name="workspace-specific",
            workspaces=["ws-prod-1", "ws-prod-2"],
        )
        assert policy.matches(workspace="ws-prod-1") is True
        assert policy.matches(workspace="ws-prod-2") is True
        assert policy.matches(workspace="ws-dev-1") is False

    def test_disabled_policy_never_matches(self):
        """Test disabled policy never matches."""
        policy = ControlPlanePolicy(
            name="disabled-policy",
            enabled=False,
        )
        assert policy.matches() is False
        assert policy.matches(task_type="any") is False

    def test_policy_region_allowed_no_constraint(self):
        """Test region check without constraint."""
        policy = ControlPlanePolicy(name="no-region-constraint")
        assert policy.is_region_allowed("us-east-1") is True
        assert policy.is_region_allowed("any-region") is True

    def test_policy_region_allowed_with_constraint(self):
        """Test region check with constraint."""
        policy = ControlPlanePolicy(
            name="region-constrained",
            region_constraint=RegionConstraint(
                allowed_regions=["us-east-1", "us-west-2"],
            ),
        )
        assert policy.is_region_allowed("us-east-1") is True
        assert policy.is_region_allowed("eu-west-1") is False

    def test_policy_versioning(self):
        """Test policy versioning fields."""
        policy = ControlPlanePolicy(
            name="versioned-policy",
            version=3,
            updated_by="admin@example.com",
            previous_version_id="policy_abc123",
        )
        assert policy.version == 3
        assert policy.updated_by == "admin@example.com"
        assert policy.previous_version_id == "policy_abc123"


class TestPolicyEnums:
    """Tests for policy enums."""

    def test_policy_scope_values(self):
        """Test PolicyScope enum values."""
        assert PolicyScope.GLOBAL.value == "global"
        assert PolicyScope.TASK_TYPE.value == "task_type"
        assert PolicyScope.CAPABILITY.value == "capability"
        assert PolicyScope.REGION.value == "region"
        assert PolicyScope.WORKSPACE.value == "workspace"

    def test_enforcement_level_values(self):
        """Test EnforcementLevel enum values."""
        assert EnforcementLevel.WARN.value == "warn"
        assert EnforcementLevel.SOFT.value == "soft"
        assert EnforcementLevel.HARD.value == "hard"

    def test_policy_decision_values(self):
        """Test PolicyDecision enum values."""
        assert PolicyDecision.ALLOW.value == "allow"
        assert PolicyDecision.DENY.value == "deny"
        assert PolicyDecision.WARN.value == "warn"
        assert PolicyDecision.ESCALATE.value == "escalate"


class TestPolicyViolationError:
    """Tests for PolicyViolationError exception."""

    def test_exception_message(self):
        """Test exception message format."""
        result = PolicyEvaluationResult(
            allowed=False,
            decision=PolicyDecision.DENY,
            enforcement_level=EnforcementLevel.HARD,
            reason="Agent not in allowlist",
            policy_id="policy_123",
            policy_name="restrict-agents",
        )
        error = PolicyViolationError(
            result=result,
            task_type="production-deploy",
            agent_id="untrusted-agent",
            region="us-east-1",
        )
        assert "Policy violation" in str(error)
        assert "hard" in str(error)
        assert error.task_type == "production-deploy"
        assert error.agent_id == "untrusted-agent"
        assert error.region == "us-east-1"
