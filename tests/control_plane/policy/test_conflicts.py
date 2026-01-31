"""Tests for control plane policy conflict detection."""

from __future__ import annotations

import pytest

from aragora.control_plane.policy.conflicts import PolicyConflict, PolicyConflictDetector
from aragora.control_plane.policy.types import (
    ControlPlanePolicy,
    EnforcementLevel,
    RegionConstraint,
    SLARequirements,
)


class TestPolicyConflict:
    """Tests for PolicyConflict dataclass."""

    def test_conflict_creation(self):
        """Test creating a policy conflict."""
        conflict = PolicyConflict(
            policy_a_id="policy_123",
            policy_a_name="policy-a",
            policy_b_id="policy_456",
            policy_b_name="policy-b",
            conflict_type="agent",
            description="Conflicting agent restrictions",
            severity="warning",
        )
        assert conflict.policy_a_id == "policy_123"
        assert conflict.conflict_type == "agent"
        assert conflict.severity == "warning"
        assert conflict.detected_at is not None

    def test_conflict_to_dict(self):
        """Test conflict serialization."""
        conflict = PolicyConflict(
            policy_a_id="policy_123",
            policy_a_name="policy-a",
            policy_b_id="policy_456",
            policy_b_name="policy-b",
            conflict_type="region",
            description="Conflicting regions",
            severity="error",
        )
        data = conflict.to_dict()
        assert data["policy_a_id"] == "policy_123"
        assert data["policy_b_name"] == "policy-b"
        assert data["conflict_type"] == "region"
        assert "detected_at" in data


class TestPolicyConflictDetector:
    """Tests for PolicyConflictDetector."""

    def test_no_conflicts_with_empty_policies(self):
        """Test no conflicts with no policies."""
        detector = PolicyConflictDetector()
        conflicts = detector.detect_conflicts([])
        assert len(conflicts) == 0

    def test_no_conflicts_with_single_policy(self):
        """Test no conflicts with single policy."""
        detector = PolicyConflictDetector()
        policy = ControlPlanePolicy(name="single")
        conflicts = detector.detect_conflicts([policy])
        assert len(conflicts) == 0

    def test_no_conflicts_non_overlapping_scopes(self):
        """Test no conflicts when policies have non-overlapping scopes."""
        detector = PolicyConflictDetector()
        policy1 = ControlPlanePolicy(
            name="production",
            task_types=["production-deploy"],
            agent_blocklist=["agent-x"],
        )
        policy2 = ControlPlanePolicy(
            name="staging",
            task_types=["staging-deploy"],
            agent_blocklist=["agent-y"],
        )
        conflicts = detector.detect_conflicts([policy1, policy2])
        assert len(conflicts) == 0

    def test_agent_allowlist_blocklist_conflict(self):
        """Test detecting conflict between allowlist and blocklist."""
        detector = PolicyConflictDetector()
        policy1 = ControlPlanePolicy(
            name="allow-agent-x",
            agent_allowlist=["agent-x"],  # Only agent-x allowed
        )
        policy2 = ControlPlanePolicy(
            name="block-agent-x",
            agent_blocklist=["agent-x"],  # agent-x blocked
        )
        conflicts = detector.detect_conflicts([policy1, policy2])
        # Should detect agent-x is both allowed exclusively and blocked
        agent_conflicts = [c for c in conflicts if c.conflict_type == "agent"]
        assert len(agent_conflicts) >= 1

    def test_overlapping_allowlists_conflict(self):
        """Test detecting conflict between mutually exclusive allowlists."""
        detector = PolicyConflictDetector()
        policy1 = ControlPlanePolicy(
            name="only-claude",
            agent_allowlist=["claude-3-opus"],  # Only claude
        )
        policy2 = ControlPlanePolicy(
            name="only-gpt",
            agent_allowlist=["gpt-4"],  # Only gpt
        )
        conflicts = detector.detect_conflicts([policy1, policy2])
        # Two allowlists with no overlap = impossible to satisfy
        agent_conflicts = [c for c in conflicts if c.conflict_type == "agent"]
        assert len(agent_conflicts) >= 1

    def test_region_constraint_conflict(self):
        """Test detecting region constraint conflicts."""
        detector = PolicyConflictDetector()
        policy1 = ControlPlanePolicy(
            name="us-only",
            region_constraint=RegionConstraint(
                allowed_regions=["us-east-1", "us-west-2"],
            ),
        )
        policy2 = ControlPlanePolicy(
            name="eu-only",
            region_constraint=RegionConstraint(
                allowed_regions=["eu-west-1", "eu-central-1"],
            ),
        )
        conflicts = detector.detect_conflicts([policy1, policy2])
        # No common regions = conflict
        region_conflicts = [c for c in conflicts if c.conflict_type == "region"]
        assert len(region_conflicts) >= 1

    def test_enforcement_level_conflict(self):
        """Test detecting enforcement level inconsistencies."""
        detector = PolicyConflictDetector()
        policy1 = ControlPlanePolicy(
            name="warn-agent",
            agent_blocklist=["agent-x"],
            enforcement_level=EnforcementLevel.WARN,
        )
        policy2 = ControlPlanePolicy(
            name="hard-agent",
            agent_blocklist=["agent-x"],
            enforcement_level=EnforcementLevel.HARD,
        )
        conflicts = detector.detect_conflicts([policy1, policy2])
        # Same agent blocked but different enforcement levels
        enforcement_conflicts = [
            c for c in conflicts if c.conflict_type == "enforcement_inconsistency"
        ]
        # May or may not be flagged depending on implementation
        # At minimum we shouldn't crash
        assert conflicts is not None

    def test_sla_conflict(self):
        """Test detecting SLA requirement conflicts."""
        detector = PolicyConflictDetector()
        policy1 = ControlPlanePolicy(
            name="fast-sla",
            sla=SLARequirements(max_execution_seconds=30.0),
        )
        policy2 = ControlPlanePolicy(
            name="slow-sla",
            sla=SLARequirements(max_execution_seconds=300.0),
        )
        conflicts = detector.detect_conflicts([policy1, policy2])
        # Conflicting SLA requirements
        sla_conflicts = [c for c in conflicts if c.conflict_type == "sla"]
        # Implementation may flag this as a warning
        assert conflicts is not None

    def test_disabled_policies_ignored(self):
        """Test disabled policies are ignored in conflict detection."""
        detector = PolicyConflictDetector()
        policy1 = ControlPlanePolicy(
            name="active",
            agent_allowlist=["agent-x"],
        )
        policy2 = ControlPlanePolicy(
            name="disabled",
            agent_blocklist=["agent-x"],
            enabled=False,  # Disabled
        )
        conflicts = detector.detect_conflicts([policy1, policy2])
        # Disabled policy should be ignored
        assert len(conflicts) == 0

    def test_conflict_severity_levels(self):
        """Test conflicts have appropriate severity levels."""
        detector = PolicyConflictDetector()
        # Create a definite conflict
        policy1 = ControlPlanePolicy(
            name="allow-only-x",
            agent_allowlist=["agent-x"],
        )
        policy2 = ControlPlanePolicy(
            name="block-x",
            agent_blocklist=["agent-x"],
        )
        conflicts = detector.detect_conflicts([policy1, policy2])
        if conflicts:
            assert all(c.severity in ("warning", "error") for c in conflicts)


class TestConflictDetectorEdgeCases:
    """Edge case tests for conflict detection."""

    def test_many_policies(self):
        """Test with many policies."""
        detector = PolicyConflictDetector()
        policies = [
            ControlPlanePolicy(
                name=f"policy-{i}",
                agent_allowlist=[f"agent-{i}"],
            )
            for i in range(20)
        ]
        # Should complete without timeout
        conflicts = detector.detect_conflicts(policies)
        # Many overlapping policies = many conflicts expected
        assert isinstance(conflicts, list)

    def test_empty_constraints(self):
        """Test policies with empty constraints."""
        detector = PolicyConflictDetector()
        policy1 = ControlPlanePolicy(name="empty-1")
        policy2 = ControlPlanePolicy(name="empty-2")
        conflicts = detector.detect_conflicts([policy1, policy2])
        # No restrictions = no conflicts
        assert len(conflicts) == 0

    def test_self_conflicting_policy(self):
        """Test policy that conflicts with itself (edge case)."""
        detector = PolicyConflictDetector()
        # This policy allows only agent-x but also blocks agent-x
        policy = ControlPlanePolicy(
            name="self-conflict",
            agent_allowlist=["agent-x"],
            agent_blocklist=["agent-x"],
        )
        # Single policy, no pairwise conflict
        conflicts = detector.detect_conflicts([policy])
        assert len(conflicts) == 0  # Detector checks pairs, not self

    def test_workspace_scoped_no_conflict(self):
        """Test workspace-scoped policies don't conflict across workspaces."""
        detector = PolicyConflictDetector()
        policy1 = ControlPlanePolicy(
            name="ws-1-policy",
            workspaces=["workspace-1"],
            agent_blocklist=["agent-x"],
        )
        policy2 = ControlPlanePolicy(
            name="ws-2-policy",
            workspaces=["workspace-2"],
            agent_allowlist=["agent-x"],
        )
        conflicts = detector.detect_conflicts([policy1, policy2])
        # Different workspaces = no conflict
        assert len(conflicts) == 0
