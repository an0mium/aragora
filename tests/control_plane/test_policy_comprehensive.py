"""
Comprehensive tests for Control Plane Policy System.

This test module provides extensive coverage of the policy system including:
- Policy types and dataclasses
- Policy conflict detection (all conflict types)
- Policy evaluation logic (all enforcement paths)
- Policy caching behavior (Redis integration)
- Policy sync scheduling
- Policy history and versioning
- Policy sync from compliance store
- Factory functions
- Edge cases and error handling

Target: 50+ test functions with good coverage of all policy operations.

Run with:
    pytest tests/control_plane/test_policy_comprehensive.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.control_plane.policy import (
    ControlPlanePolicy,
    ControlPlanePolicyManager,
    EnforcementLevel,
    PolicyConflict,
    PolicyConflictDetector,
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyHistory,
    PolicyScope,
    PolicySyncScheduler,
    PolicyVersion,
    PolicyViolation,
    PolicyViolationError,
    RedisPolicyCache,
    RegionConstraint,
    SLARequirements,
    create_agent_tier_policy,
    create_production_policy,
    create_sensitive_data_policy,
    create_sla_policy,
    get_policy_history,
)
from aragora.control_plane.policy.sync import PolicyStoreSync


# =============================================================================
# PolicyViolationError Tests
# =============================================================================


class TestPolicyViolationError:
    """Tests for PolicyViolationError exception class."""

    def test_error_message_includes_enforcement_level(self):
        """Test error message includes enforcement level."""
        result = PolicyEvaluationResult(
            decision=PolicyDecision.DENY,
            allowed=False,
            policy_id="test-policy",
            policy_name="Test Policy",
            reason="Agent blocked",
            enforcement_level=EnforcementLevel.HARD,
        )
        error = PolicyViolationError(result)

        assert "hard" in str(error).lower()
        assert "Agent blocked" in str(error)

    def test_error_preserves_context(self):
        """Test error preserves task context."""
        result = PolicyEvaluationResult(
            decision=PolicyDecision.DENY,
            allowed=False,
            policy_id="test-policy",
            policy_name="Test Policy",
            reason="Denied",
            enforcement_level=EnforcementLevel.SOFT,
        )
        error = PolicyViolationError(
            result,
            task_type="analysis",
            agent_id="gpt-4",
            region="us-east-1",
        )

        assert error.task_type == "analysis"
        assert error.agent_id == "gpt-4"
        assert error.region == "us-east-1"
        assert error.result is result

    def test_error_with_warn_level(self):
        """Test error with WARN enforcement level."""
        result = PolicyEvaluationResult(
            decision=PolicyDecision.WARN,
            allowed=True,
            policy_id="warn-policy",
            policy_name="Warning Policy",
            reason="Soft violation",
            enforcement_level=EnforcementLevel.WARN,
        )
        error = PolicyViolationError(result)

        assert "warn" in str(error).lower()


# =============================================================================
# PolicyScope Tests
# =============================================================================


class TestPolicyScope:
    """Tests for PolicyScope enum."""

    def test_all_scope_values(self):
        """Test all scope values exist."""
        assert PolicyScope.GLOBAL.value == "global"
        assert PolicyScope.TASK_TYPE.value == "task_type"
        assert PolicyScope.CAPABILITY.value == "capability"
        assert PolicyScope.REGION.value == "region"
        assert PolicyScope.WORKSPACE.value == "workspace"

    def test_scope_from_string(self):
        """Test creating scope from string value."""
        assert PolicyScope("global") == PolicyScope.GLOBAL
        assert PolicyScope("task_type") == PolicyScope.TASK_TYPE

    def test_invalid_scope_raises(self):
        """Test invalid scope value raises ValueError."""
        with pytest.raises(ValueError):
            PolicyScope("invalid_scope")


# =============================================================================
# SLARequirements Edge Cases
# =============================================================================


class TestSLARequirementsEdgeCases:
    """Additional tests for SLARequirements edge cases."""

    def test_boundary_execution_time(self):
        """Test execution time exactly at boundary."""
        sla = SLARequirements(max_execution_seconds=100.0)
        assert sla.is_execution_time_compliant(100.0) is True
        assert sla.is_execution_time_compliant(100.001) is False

    def test_boundary_queue_time(self):
        """Test queue time exactly at boundary."""
        sla = SLARequirements(max_queue_seconds=50.0)
        assert sla.is_queue_time_compliant(50.0) is True
        assert sla.is_queue_time_compliant(50.001) is False

    def test_zero_limits(self):
        """Test with zero limits."""
        sla = SLARequirements(
            max_execution_seconds=0.0,
            max_queue_seconds=0.0,
        )
        assert sla.is_execution_time_compliant(0.0) is True
        assert sla.is_execution_time_compliant(0.001) is False

    def test_from_dict_with_missing_keys(self):
        """Test from_dict uses defaults for missing keys."""
        sla = SLARequirements.from_dict({})
        assert sla.max_execution_seconds == 300.0
        assert sla.max_queue_seconds == 60.0
        assert sla.min_agents_available == 1

    def test_from_dict_with_partial_data(self):
        """Test from_dict with only some values provided."""
        sla = SLARequirements.from_dict({"max_execution_seconds": 120.0})
        assert sla.max_execution_seconds == 120.0
        assert sla.max_queue_seconds == 60.0  # Default


# =============================================================================
# RegionConstraint Edge Cases
# =============================================================================


class TestRegionConstraintEdgeCases:
    """Additional tests for RegionConstraint edge cases."""

    def test_empty_allowlist_allows_all(self):
        """Test empty allowlist allows all regions."""
        constraint = RegionConstraint(allowed_regions=[])
        assert constraint.is_region_allowed("any-region") is True

    def test_data_residency_without_data_region(self):
        """Test data residency check without data region specified."""
        constraint = RegionConstraint(require_data_residency=True)
        # Without data_region, should allow
        assert constraint.is_region_allowed("us-east-1") is True

    def test_cross_region_disabled_with_data_region(self):
        """Test cross-region check with allow_cross_region=False."""
        constraint = RegionConstraint(
            require_data_residency=True,
            allow_cross_region=False,
        )
        assert constraint.is_region_allowed("us-east-1", data_region="us-east-1") is True
        assert constraint.is_region_allowed("eu-west-1", data_region="us-east-1") is False

    def test_blocklist_empty(self):
        """Test empty blocklist blocks nothing."""
        constraint = RegionConstraint(blocked_regions=[])
        assert constraint.is_region_allowed("any-region") is True

    def test_from_dict_empty(self):
        """Test from_dict with empty dict."""
        constraint = RegionConstraint.from_dict({})
        assert constraint.allowed_regions == []
        assert constraint.blocked_regions == []
        assert constraint.require_data_residency is False


# =============================================================================
# ControlPlanePolicy Edge Cases
# =============================================================================


class TestControlPlanePolicyEdgeCases:
    """Additional tests for ControlPlanePolicy edge cases."""

    def test_matches_with_no_filters(self):
        """Test policy with no filters matches everything."""
        policy = ControlPlanePolicy(name="no-filters")
        assert policy.matches(task_type="any", capability="any", workspace="any") is True

    def test_matches_with_none_values(self):
        """Test matches handles None values correctly."""
        policy = ControlPlanePolicy(name="test", task_types=["task-a"])
        # None task_type should not be filtered
        assert policy.matches(task_type=None) is True

    def test_matches_capability_in_list(self):
        """Test capability matching when in list."""
        policy = ControlPlanePolicy(name="test", capabilities=["cap-a", "cap-b"])
        assert policy.matches(capability="cap-a") is True
        assert policy.matches(capability="cap-c") is False

    def test_is_agent_allowed_empty_lists(self):
        """Test agent allowed with both lists empty."""
        policy = ControlPlanePolicy(name="test")
        assert policy.is_agent_allowed("any-agent") is True

    def test_policy_id_auto_generated(self):
        """Test policy ID is auto-generated when not provided."""
        policy = ControlPlanePolicy(name="test")
        assert policy.id.startswith("policy_")
        assert len(policy.id) > 8

    def test_created_at_auto_set(self):
        """Test created_at is automatically set."""
        before = datetime.now(timezone.utc)
        policy = ControlPlanePolicy(name="test")
        after = datetime.now(timezone.utc)

        assert before <= policy.created_at <= after

    def test_from_dict_with_iso_timestamp(self):
        """Test from_dict handles ISO timestamp string."""
        ts = datetime.now(timezone.utc).isoformat()
        data = {"name": "test", "created_at": ts}
        policy = ControlPlanePolicy.from_dict(data)
        assert isinstance(policy.created_at, datetime)

    def test_from_dict_with_none_timestamp(self):
        """Test from_dict handles None timestamp."""
        data = {"name": "test", "created_at": None}
        policy = ControlPlanePolicy.from_dict(data)
        assert isinstance(policy.created_at, datetime)

    def test_to_dict_includes_all_fields(self):
        """Test to_dict includes all required fields."""
        policy = ControlPlanePolicy(
            name="full-policy",
            description="Description",
            scope=PolicyScope.REGION,
            task_types=["task-a"],
            capabilities=["cap-a"],
            workspaces=["ws-1"],
            agent_allowlist=["agent-1"],
            agent_blocklist=["agent-2"],
            region_constraint=RegionConstraint(allowed_regions=["us-east-1"]),
            sla=SLARequirements(max_execution_seconds=60.0),
            enforcement_level=EnforcementLevel.SOFT,
            enabled=True,
            priority=50,
            version=2,
            metadata={"key": "value"},
        )

        data = policy.to_dict()

        assert data["name"] == "full-policy"
        assert data["scope"] == "region"
        assert data["enforcement_level"] == "soft"
        assert data["region_constraint"]["allowed_regions"] == ["us-east-1"]
        assert data["sla"]["max_execution_seconds"] == 60.0


# =============================================================================
# PolicyManager Advanced Tests
# =============================================================================


class TestPolicyManagerAdvanced:
    """Advanced tests for ControlPlanePolicyManager."""

    @pytest.fixture
    def manager(self):
        """Create a policy manager for testing."""
        return ControlPlanePolicyManager()

    def test_evaluate_with_capability_matching(self, manager):
        """Test evaluation with capability-level policy matching."""
        policy = ControlPlanePolicy(
            name="cap-policy",
            capabilities=["code-review"],
            agent_blocklist=["weak-agent"],
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="task-a",
            agent_id="weak-agent",
            region="us-east-1",
            capabilities=["code-review"],
        )

        assert result.decision == PolicyDecision.DENY

    def test_evaluate_no_capability_match(self, manager):
        """Test evaluation when capabilities don't match policy.

        Note: The policy manager uses OR logic - if a policy matches on task_type
        OR capability, it will be evaluated. Since the policy has no task_types
        specified, it applies to all tasks, but only evaluates capability match
        when capabilities are provided.
        """
        policy = ControlPlanePolicy(
            name="cap-policy",
            capabilities=["code-review"],
            task_types=["specific-task"],  # Only apply to specific tasks
            agent_blocklist=["weak-agent"],
        )
        manager.add_policy(policy)

        # Different task type AND different capability - policy shouldn't apply
        result = manager.evaluate_task_dispatch(
            task_type="other-task",
            agent_id="weak-agent",
            region="us-east-1",
            capabilities=["debate"],
        )

        assert result.decision == PolicyDecision.ALLOW

    def test_evaluate_data_residency_violation(self, manager):
        """Test evaluation with data residency violation."""
        policy = ControlPlanePolicy(
            name="residency-policy",
            region_constraint=RegionConstraint(
                require_data_residency=True,
            ),
        )
        manager.add_policy(policy)

        result = manager.evaluate_task_dispatch(
            task_type="task",
            agent_id="agent",
            region="eu-west-1",
            data_region="us-east-1",
        )

        assert result.decision == PolicyDecision.DENY
        assert "eu-west-1" in result.reason

    def test_evaluate_sla_no_policy_found(self, manager):
        """Test SLA evaluation when policy doesn't exist."""
        result = manager.evaluate_sla_compliance(
            policy_id="nonexistent-policy",
            execution_seconds=100.0,
        )

        assert result.decision == PolicyDecision.ALLOW
        assert "No SLA requirements" in result.reason

    def test_evaluate_sla_no_sla_requirements(self, manager):
        """Test SLA evaluation when policy has no SLA."""
        policy = ControlPlanePolicy(name="no-sla-policy", sla=None)
        manager.add_policy(policy)

        result = manager.evaluate_sla_compliance(
            policy_id=policy.id,
            execution_seconds=1000.0,
        )

        assert result.decision == PolicyDecision.ALLOW

    def test_evaluate_sla_multiple_violations(self, manager):
        """Test SLA evaluation with multiple violations."""
        policy = ControlPlanePolicy(
            name="strict-sla",
            sla=SLARequirements(
                max_execution_seconds=60.0,
                max_queue_seconds=10.0,
                min_agents_available=5,
            ),
        )
        manager.add_policy(policy)

        result = manager.evaluate_sla_compliance(
            policy_id=policy.id,
            execution_seconds=120.0,
            queue_seconds=30.0,
            available_agents=2,
        )

        assert result.decision == PolicyDecision.DENY
        assert "Execution time" in result.sla_violation
        assert "Queue time" in result.sla_violation
        assert "Available agents" in result.sla_violation

    def test_violation_callback_error_handling(self):
        """Test violation callback error is handled gracefully."""

        def bad_callback(violation):
            raise RuntimeError("Callback error")

        manager = ControlPlanePolicyManager(violation_callback=bad_callback)
        policy = ControlPlanePolicy(name="test", agent_blocklist=["agent"])
        manager.add_policy(policy)

        # Should not raise
        result = manager.evaluate_task_dispatch(
            task_type="task",
            agent_id="agent",
            region="region",
        )

        assert result.decision == PolicyDecision.DENY

    def test_get_violations_with_all_filters(self, manager):
        """Test get_violations with all filters applied."""
        policy = ControlPlanePolicy(
            name="test",
            agent_blocklist=["agent-a"],
        )
        manager.add_policy(policy)

        # Generate some violations
        manager.evaluate_task_dispatch(
            task_type="task",
            agent_id="agent-a",
            region="region",
            workspace="ws-1",
        )

        violations = manager.get_violations(
            policy_id=policy.id,
            violation_type="agent",
            workspace_id="ws-1",
            limit=10,
        )

        assert len(violations) == 1

    def test_list_policies_with_all_filters(self, manager):
        """Test list_policies with all filters applied."""
        manager.add_policy(
            ControlPlanePolicy(
                name="enabled-ws1",
                enabled=True,
                task_types=["task-a"],
                workspaces=["ws-1"],
            )
        )
        manager.add_policy(
            ControlPlanePolicy(
                name="disabled",
                enabled=False,
            )
        )

        policies = manager.list_policies(
            enabled_only=True,
            task_type="task-a",
            workspace="ws-1",
        )

        assert len(policies) == 1
        assert policies[0].name == "enabled-ws1"

    def test_metrics_after_multiple_operations(self, manager):
        """Test metrics after various operations."""
        policy = ControlPlanePolicy(
            name="metrics-test",
            agent_blocklist=["blocked"],
            enforcement_level=EnforcementLevel.WARN,
        )
        manager.add_policy(policy)

        # Perform various evaluations
        manager.evaluate_task_dispatch("task", "allowed", "region")
        manager.evaluate_task_dispatch("task", "blocked", "region")
        manager.evaluate_task_dispatch("task", "allowed", "region")

        metrics = manager.get_metrics()

        assert metrics["evaluations"] == 3
        assert metrics["allowed"] >= 2
        assert metrics["warned"] >= 1


# =============================================================================
# PolicyConflictDetector Advanced Tests
# =============================================================================


class TestPolicyConflictDetectorAdvanced:
    """Advanced tests for PolicyConflictDetector."""

    @pytest.fixture
    def detector(self):
        """Create a conflict detector for testing."""
        return PolicyConflictDetector()

    def test_no_conflict_different_task_types(self, detector):
        """Test no conflict when task types don't overlap."""
        policies = [
            ControlPlanePolicy(
                name="task-a-policy",
                task_types=["task-a"],
                agent_allowlist=["agent-1"],
            ),
            ControlPlanePolicy(
                name="task-b-policy",
                task_types=["task-b"],
                agent_allowlist=["agent-2"],  # Different agents, but different scope
            ),
        ]

        conflicts = detector.detect_conflicts(policies)
        assert len(conflicts) == 0

    def test_no_conflict_different_workspaces(self, detector):
        """Test no conflict when workspaces don't overlap."""
        policies = [
            ControlPlanePolicy(
                name="ws-1-policy",
                workspaces=["ws-1"],
                agent_allowlist=["agent-1"],
            ),
            ControlPlanePolicy(
                name="ws-2-policy",
                workspaces=["ws-2"],
                agent_allowlist=["agent-2"],
            ),
        ]

        conflicts = detector.detect_conflicts(policies)
        assert len(conflicts) == 0

    def test_conflict_with_overlapping_task_types(self, detector):
        """Test conflict detection with overlapping task types."""
        policies = [
            ControlPlanePolicy(
                name="policy-a",
                task_types=["task-a", "task-b"],
                agent_allowlist=["agent-1"],
            ),
            ControlPlanePolicy(
                name="policy-b",
                task_types=["task-b", "task-c"],
                agent_allowlist=["agent-2"],  # Non-overlapping agents
            ),
        ]

        conflicts = detector.detect_conflicts(policies)
        # Should detect conflict because task-b overlaps and no agent satisfies both
        assert len(conflicts) >= 1

    def test_region_allowlist_blocklist_conflict(self, detector):
        """Test region conflict when allowlist intersects with blocklist.

        The conflict detector checks if policy A's allowed_regions intersects
        with policy B's blocked_regions, detecting that a region is simultaneously
        allowed by one policy and blocked by another.
        """
        policies = [
            ControlPlanePolicy(
                name="allow-us",
                region_constraint=RegionConstraint(allowed_regions=["us-east-1", "us-west-2"]),
            ),
            ControlPlanePolicy(
                name="block-us",
                region_constraint=RegionConstraint(blocked_regions=["us-east-1"]),
            ),
        ]

        conflicts = detector.detect_conflicts(policies)
        region_conflicts = [c for c in conflicts if c.conflict_type == "region"]
        # allow-us allows us-east-1, block-us blocks it - this is a conflict
        assert len(region_conflicts) == 1
        assert "us-east-1" in str(region_conflicts[0].description)

    def test_sla_response_time_conflict(self, detector):
        """Test SLA response time conflict detection."""
        policies = [
            ControlPlanePolicy(
                name="fast-response",
                task_types=["api"],
                sla=SLARequirements(response_time_p99_ms=100.0),
            ),
            ControlPlanePolicy(
                name="slow-response",
                task_types=["api"],
                sla=SLARequirements(response_time_p99_ms=1000.0),  # 10x difference
            ),
        ]

        conflicts = detector.detect_conflicts(policies)
        sla_conflicts = [c for c in conflicts if c.conflict_type == "sla_response_time"]
        assert len(sla_conflicts) >= 1

    def test_multiple_conflict_types_same_policies(self, detector):
        """Test detecting multiple conflict types between same policies."""
        policies = [
            ControlPlanePolicy(
                name="policy-a",
                task_types=["task"],
                agent_allowlist=["agent-a"],
                region_constraint=RegionConstraint(allowed_regions=["us-east-1"]),
            ),
            ControlPlanePolicy(
                name="policy-b",
                task_types=["task"],
                agent_allowlist=["agent-b"],  # Non-overlapping agents
                region_constraint=RegionConstraint(
                    allowed_regions=["eu-west-1"]
                ),  # Non-overlapping regions
            ),
        ]

        conflicts = detector.detect_conflicts(policies)
        agent_conflicts = [c for c in conflicts if c.conflict_type == "agent"]
        region_conflicts = [c for c in conflicts if c.conflict_type == "region"]

        assert len(agent_conflicts) >= 1
        assert len(region_conflicts) >= 1

    def test_conflict_severity_based_on_enforcement(self, detector):
        """Test conflict severity is based on enforcement level."""
        # Hard enforcement -> error severity
        policies_hard = [
            ControlPlanePolicy(
                name="allow",
                agent_allowlist=["agent-a"],
                enforcement_level=EnforcementLevel.HARD,
            ),
            ControlPlanePolicy(
                name="block",
                agent_blocklist=["agent-a"],
                enforcement_level=EnforcementLevel.HARD,
            ),
        ]

        conflicts_hard = detector.detect_conflicts(policies_hard)
        assert conflicts_hard[0].severity == "error"

        # Warn enforcement -> warning severity
        policies_warn = [
            ControlPlanePolicy(
                name="allow",
                agent_allowlist=["agent-a"],
                enforcement_level=EnforcementLevel.HARD,
            ),
            ControlPlanePolicy(
                name="block",
                agent_blocklist=["agent-a"],
                enforcement_level=EnforcementLevel.WARN,
            ),
        ]

        conflicts_warn = detector.detect_conflicts(policies_warn)
        assert conflicts_warn[0].severity == "warning"

    def test_empty_policies_list(self, detector):
        """Test with empty policies list."""
        conflicts = detector.detect_conflicts([])
        assert conflicts == []

    def test_single_policy_no_conflicts(self, detector):
        """Test single policy never has conflicts."""
        policies = [
            ControlPlanePolicy(name="only-policy", agent_allowlist=["agent"]),
        ]
        conflicts = detector.detect_conflicts(policies)
        assert conflicts == []


# =============================================================================
# RedisPolicyCache Advanced Tests
# =============================================================================


class TestRedisPolicyCacheAdvanced:
    """Advanced tests for RedisPolicyCache."""

    def test_cache_key_uniqueness(self):
        """Test cache keys are unique for different inputs."""
        cache = RedisPolicyCache()
        keys = set()

        combinations = [
            ("task-a", "agent-1", "us-east-1", None),
            ("task-b", "agent-1", "us-east-1", None),
            ("task-a", "agent-2", "us-east-1", None),
            ("task-a", "agent-1", "eu-west-1", None),
            ("task-a", "agent-1", "us-east-1", "ws-1"),
        ]

        for task, agent, region, workspace in combinations:
            key = cache._make_cache_key(task, agent, region, workspace)
            keys.add(key)

        assert len(keys) == len(combinations)

    def test_cache_key_with_policy_version(self):
        """Test cache key changes with policy version."""
        cache = RedisPolicyCache()

        key_v1 = cache._make_cache_key("task", "agent", "region", None, "v1")
        key_v2 = cache._make_cache_key("task", "agent", "region", None, "v2")

        assert key_v1 != key_v2

    @pytest.mark.asyncio
    async def test_close_without_connection(self):
        """Test close when not connected."""
        cache = RedisPolicyCache()
        # Should not raise
        await cache.close()

    @pytest.mark.asyncio
    async def test_invalidate_all_without_connection(self):
        """Test invalidate_all when not connected."""
        cache = RedisPolicyCache()
        deleted = await cache.invalidate_all()
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_cache_with_mocked_redis(self):
        """Test cache operations with mocked Redis."""
        cache = RedisPolicyCache()

        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        cache._redis = mock_redis

        # Test get (miss)
        result = await cache.get("task", "agent", "region")
        assert result is None
        assert cache._stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_cache_hit_deserialization(self):
        """Test cache hit deserializes correctly."""
        cache = RedisPolicyCache()

        # Prepare cached data
        cached_result = PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW,
            allowed=True,
            policy_id="cached-policy",
            policy_name="Cached",
            reason="From cache",
            enforcement_level=EnforcementLevel.HARD,
            task_type="task",
            agent_id="agent",
            region="region",
        )
        cached_json = json.dumps(cached_result.to_dict())

        # Mock Redis
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=cached_json)
        cache._redis = mock_redis

        result = await cache.get("task", "agent", "region")

        assert result is not None
        assert result.policy_id == "cached-policy"
        assert result.decision == PolicyDecision.ALLOW
        assert cache._stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_cache_set_success(self):
        """Test successful cache set."""
        cache = RedisPolicyCache()

        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock()
        cache._redis = mock_redis

        result = PolicyEvaluationResult(
            decision=PolicyDecision.DENY,
            allowed=False,
            policy_id="test",
            policy_name="Test",
            reason="Denied",
            enforcement_level=EnforcementLevel.HARD,
        )

        success = await cache.set(result, "task", "agent", "region")

        assert success is True
        assert cache._stats["sets"] == 1
        mock_redis.setex.assert_called_once()


# =============================================================================
# PolicySyncScheduler Advanced Tests
# =============================================================================


class TestPolicySyncSchedulerAdvanced:
    """Advanced tests for PolicySyncScheduler."""

    @pytest.fixture
    def manager(self):
        """Create a policy manager."""
        return ControlPlanePolicyManager()

    @pytest.mark.asyncio
    async def test_double_start_warning(self, manager):
        """Test starting scheduler twice logs warning."""
        scheduler = PolicySyncScheduler(
            policy_manager=manager,
            sync_interval_seconds=1.0,
            sync_from_compliance_store=False,
            sync_from_control_plane_store=False,
        )

        await scheduler.start()
        await scheduler.start()  # Second start should warn

        assert scheduler.get_status()["running"] is True

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, manager):
        """Test stopping scheduler that wasn't started."""
        scheduler = PolicySyncScheduler(
            policy_manager=manager,
            sync_interval_seconds=1.0,
        )

        # Should not raise
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_sync_detects_changes(self, manager):
        """Test sync detects policy changes."""
        scheduler = PolicySyncScheduler(
            policy_manager=manager,
            sync_interval_seconds=1.0,
            sync_from_compliance_store=False,
            sync_from_control_plane_store=False,
        )

        # First sync - establish baseline
        result1 = await scheduler.sync_now()
        assert result1["changes_detected"] is False

        # Add policy
        manager.add_policy(ControlPlanePolicy(name="new-policy"))

        # Second sync - should detect change
        result2 = await scheduler.sync_now()
        assert result2["changes_detected"] is True

    @pytest.mark.asyncio
    async def test_sync_with_workspace_filter(self, manager):
        """Test sync with workspace filter."""
        scheduler = PolicySyncScheduler(
            policy_manager=manager,
            sync_interval_seconds=1.0,
            workspace_id="specific-workspace",
            sync_from_compliance_store=False,
            sync_from_control_plane_store=False,
        )

        status = scheduler.get_status()
        assert status["workspace_id"] == "specific-workspace"

    @pytest.mark.asyncio
    async def test_conflict_callback_error_handling(self, manager):
        """Test conflict callback errors are handled."""

        def bad_callback(conflicts):
            raise RuntimeError("Callback error")

        scheduler = PolicySyncScheduler(
            policy_manager=manager,
            sync_interval_seconds=1.0,
            conflict_callback=bad_callback,
            sync_from_compliance_store=False,
            sync_from_control_plane_store=False,
        )

        # Add conflicting policies
        manager.add_policy(ControlPlanePolicy(name="a", agent_allowlist=["x"]))
        manager.add_policy(ControlPlanePolicy(name="b", agent_blocklist=["x"]))

        # Should not raise
        result = await scheduler.sync_now()
        assert result["conflicts_detected"] > 0

    @pytest.mark.asyncio
    async def test_policy_version_property(self, manager):
        """Test policy_version property."""
        scheduler = PolicySyncScheduler(
            policy_manager=manager,
            sync_interval_seconds=1.0,
            sync_from_compliance_store=False,
            sync_from_control_plane_store=False,
        )

        assert scheduler.policy_version is None

        await scheduler.sync_now()

        assert scheduler.policy_version is not None
        assert len(scheduler.policy_version) == 16  # SHA256 truncated to 16 chars


# =============================================================================
# PolicyHistory Tests
# =============================================================================


class TestPolicyHistoryAdvanced:
    """Tests for PolicyHistory."""

    @pytest.fixture
    def history(self):
        """Create a policy history instance."""
        return PolicyHistory(max_versions_per_policy=5)

    @pytest.mark.asyncio
    async def test_record_version(self, history):
        """Test recording a policy version."""
        policy = ControlPlanePolicy(name="test-policy", version=1)

        version = await history.record_version(
            policy,
            change_description="Initial creation",
            changed_by="admin",
        )

        assert version.policy_id == policy.id
        assert version.version == 1
        assert version.change_description == "Initial creation"
        assert version.created_by == "admin"

    @pytest.mark.asyncio
    async def test_get_history(self, history):
        """Test getting version history."""
        policy_id = "test-policy-1"

        # Record multiple versions
        for i in range(1, 4):
            policy = ControlPlanePolicy(
                id=policy_id,
                name="test",
                version=i,
            )
            await history.record_version(policy, change_description=f"Version {i}")

        versions = await history.get_history(policy_id)

        assert len(versions) == 3
        # Newest first
        assert versions[0].version == 3
        assert versions[-1].version == 1

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self, history):
        """Test getting limited version history."""
        policy_id = "test-policy-2"

        for i in range(1, 6):
            policy = ControlPlanePolicy(id=policy_id, name="test", version=i)
            await history.record_version(policy)

        versions = await history.get_history(policy_id, limit=2)

        assert len(versions) == 2
        assert versions[0].version == 5

    @pytest.mark.asyncio
    async def test_get_specific_version(self, history):
        """Test getting a specific version."""
        policy_id = "test-policy-3"

        for i in range(1, 4):
            policy = ControlPlanePolicy(id=policy_id, name="test", version=i)
            await history.record_version(policy)

        version = await history.get_version(policy_id, 2)

        assert version is not None
        assert version.version == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent_version(self, history):
        """Test getting version that doesn't exist."""
        version = await history.get_version("nonexistent-policy", 1)
        assert version is None

    @pytest.mark.asyncio
    async def test_max_versions_pruning(self, history):
        """Test that old versions are pruned."""
        policy_id = "pruned-policy"

        # Record more than max_versions
        for i in range(1, 10):
            policy = ControlPlanePolicy(id=policy_id, name="test", version=i)
            await history.record_version(policy)

        versions = await history.get_history(policy_id, limit=100)

        assert len(versions) == 5  # max_versions_per_policy
        assert versions[0].version == 9
        assert versions[-1].version == 5

    @pytest.mark.asyncio
    async def test_rollback_to_version(self, history):
        """Test rolling back to a previous version.

        The rollback creates a new version based on the historical data,
        incrementing the version number and adding rollback metadata.
        """
        policy_id = "rollback-policy"

        # Create initial versions
        for i in range(1, 4):
            policy = ControlPlanePolicy(
                id=policy_id,
                name=f"Version {i}",
                version=i,
                metadata={},
            )
            await history.record_version(policy)

        # Rollback to version 1
        restored = await history.rollback_to_version(
            policy_id,
            version=1,
            rolled_back_by="admin",
        )

        assert restored is not None
        # The restored policy should have rolled_back_from_version metadata
        assert restored.metadata.get("rolled_back_from_version") == 1
        # The name should be from version 1
        assert restored.name == "Version 1"

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_version(self, history):
        """Test rollback to nonexistent version returns None."""
        restored = await history.rollback_to_version(
            "some-policy",
            version=999,
        )
        assert restored is None

    def test_get_stats(self, history):
        """Test getting history statistics."""
        stats = history.get_stats()

        assert stats["tracked_policies"] == 0
        assert stats["total_versions"] == 0
        assert stats["max_versions_per_policy"] == 5

    def test_global_policy_history(self):
        """Test global policy history singleton."""
        history1 = get_policy_history()
        history2 = get_policy_history()

        assert history1 is history2


# =============================================================================
# PolicyStoreSync Tests
# =============================================================================


class TestPolicyStoreSyncAdvanced:
    """Tests for PolicyStoreSync."""

    @pytest.fixture
    def manager(self):
        """Create a policy manager."""
        return ControlPlanePolicyManager()

    @pytest.fixture
    def sync(self, manager):
        """Create a PolicyStoreSync instance."""
        return PolicyStoreSync(manager)

    def test_clear_synced_policies(self, manager, sync):
        """Test clearing synced policies."""
        # Manually add a policy as if synced
        policy = ControlPlanePolicy(name="synced-policy")
        manager.add_policy(policy)
        sync._synced_policy_ids.add(policy.id)

        cleared = sync.clear_synced_policies()

        assert cleared == 1
        assert manager.get_policy(policy.id) is None

    def test_framework_mappings_exist(self, sync):
        """Test that framework mappings are defined."""
        assert "data_residency" in sync.FRAMEWORK_MAPPINGS
        assert "agent_restrictions" in sync.FRAMEWORK_MAPPINGS
        assert "sla_requirements" in sync.FRAMEWORK_MAPPINGS
        assert "task_restrictions" in sync.FRAMEWORK_MAPPINGS


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctionsAdvanced:
    """Additional tests for factory functions."""

    def test_create_production_policy_defaults(self):
        """Test production policy with default values."""
        policy = create_production_policy()

        assert policy.agent_allowlist == []
        assert policy.region_constraint.allowed_regions == []
        assert policy.enforcement_level == EnforcementLevel.HARD
        assert policy.priority == 100

    def test_create_sensitive_data_policy_with_all_options(self):
        """Test sensitive data policy with all options."""
        policy = create_sensitive_data_policy(
            data_regions=["us-east-1", "us-west-2"],
            blocked_regions=["cn-north-1", "ru-central-1"],
        )

        assert policy.region_constraint.allowed_regions == ["us-east-1", "us-west-2"]
        assert policy.region_constraint.blocked_regions == ["cn-north-1", "ru-central-1"]
        assert policy.region_constraint.allow_cross_region is False

    def test_create_agent_tier_policy_defaults(self):
        """Test agent tier policy with default task_types."""
        policy = create_agent_tier_policy(
            tier="standard",
            agents=["claude-3-haiku"],
        )

        assert policy.task_types == []
        assert policy.enforcement_level == EnforcementLevel.HARD

    def test_create_sla_policy_enforcement_level(self):
        """Test SLA policy uses WARN enforcement by default."""
        policy = create_sla_policy(
            name="test-sla",
            task_types=["task-a"],
        )

        assert policy.enforcement_level == EnforcementLevel.WARN


# =============================================================================
# PolicyEvaluationResult Tests
# =============================================================================


class TestPolicyEvaluationResultAdvanced:
    """Additional tests for PolicyEvaluationResult."""

    def test_result_with_sla_violation(self):
        """Test result with SLA violation info."""
        result = PolicyEvaluationResult(
            decision=PolicyDecision.DENY,
            allowed=False,
            policy_id="sla-policy",
            policy_name="SLA Policy",
            reason="SLA violation",
            enforcement_level=EnforcementLevel.HARD,
            sla_violation="Execution time exceeded",
        )

        data = result.to_dict()
        assert data["sla_violation"] == "Execution time exceeded"

    def test_result_evaluated_at_auto_set(self):
        """Test evaluated_at is automatically set."""
        before = datetime.now(timezone.utc)
        result = PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW,
            allowed=True,
            policy_id="",
            policy_name="",
            reason="Passed",
            enforcement_level=EnforcementLevel.HARD,
        )
        after = datetime.now(timezone.utc)

        assert before <= result.evaluated_at <= after

    def test_result_escalate_decision(self):
        """Test ESCALATE decision value."""
        result = PolicyEvaluationResult(
            decision=PolicyDecision.ESCALATE,
            allowed=False,
            policy_id="escalate-policy",
            policy_name="Escalation Required",
            reason="Requires human approval",
            enforcement_level=EnforcementLevel.HARD,
        )

        assert result.decision == PolicyDecision.ESCALATE
        assert result.to_dict()["decision"] == "escalate"


# =============================================================================
# PolicyConflict Tests
# =============================================================================


class TestPolicyConflictAdvanced:
    """Additional tests for PolicyConflict dataclass."""

    def test_conflict_detected_at_auto_set(self):
        """Test detected_at is automatically set."""
        conflict = PolicyConflict(
            policy_a_id="a",
            policy_a_name="A",
            policy_b_id="b",
            policy_b_name="B",
            conflict_type="agent",
            description="Test",
            severity="error",
        )

        assert isinstance(conflict.detected_at, datetime)

    def test_conflict_to_dict_complete(self):
        """Test to_dict includes all fields."""
        conflict = PolicyConflict(
            policy_a_id="policy-1",
            policy_a_name="Policy One",
            policy_b_id="policy-2",
            policy_b_name="Policy Two",
            conflict_type="sla_execution_time",
            description="SLA conflict",
            severity="warning",
        )

        data = conflict.to_dict()

        assert data["policy_a_id"] == "policy-1"
        assert data["policy_b_id"] == "policy-2"
        assert data["conflict_type"] == "sla_execution_time"
        assert data["severity"] == "warning"
        assert "detected_at" in data


# =============================================================================
# PolicyViolation Tests
# =============================================================================


class TestPolicyViolationAdvanced:
    """Additional tests for PolicyViolation dataclass."""

    def test_violation_timestamp_auto_set(self):
        """Test timestamp is automatically set."""
        violation = PolicyViolation(
            id="v-1",
            policy_id="p-1",
            policy_name="Policy",
            violation_type="agent",
            description="Test",
        )

        assert isinstance(violation.timestamp, datetime)

    def test_violation_with_all_context(self):
        """Test violation with all context fields."""
        violation = PolicyViolation(
            id="v-full",
            policy_id="p-1",
            policy_name="Full Policy",
            violation_type="sla",
            description="Full context violation",
            task_id="task-123",
            task_type="analysis",
            agent_id="slow-agent",
            region="us-east-1",
            workspace_id="ws-prod",
            enforcement_level=EnforcementLevel.SOFT,
            metadata={"severity": "high"},
        )

        data = violation.to_dict()

        assert data["task_id"] == "task-123"
        assert data["task_type"] == "analysis"
        assert data["agent_id"] == "slow-agent"
        assert data["region"] == "us-east-1"
        assert data["workspace_id"] == "ws-prod"
        assert data["enforcement_level"] == "soft"
        assert data["metadata"]["severity"] == "high"

    def test_violation_default_enforcement_level(self):
        """Test violation defaults to HARD enforcement."""
        violation = PolicyViolation(
            id="v-1",
            policy_id="p-1",
            policy_name="Policy",
            violation_type="agent",
            description="Test",
        )

        assert violation.enforcement_level == EnforcementLevel.HARD


# =============================================================================
# PolicyVersion Tests
# =============================================================================


class TestPolicyVersionAdvanced:
    """Tests for PolicyVersion dataclass."""

    def test_version_to_dict(self):
        """Test PolicyVersion serialization."""
        version = PolicyVersion(
            policy_id="p-1",
            version=3,
            policy_data={"name": "Test", "priority": 10},
            created_by="admin",
            change_description="Updated priority",
        )

        data = version.to_dict()

        assert data["policy_id"] == "p-1"
        assert data["version"] == 3
        assert data["policy_data"]["name"] == "Test"
        assert data["created_by"] == "admin"
        assert data["change_description"] == "Updated priority"
        assert "created_at" in data

    def test_version_defaults(self):
        """Test PolicyVersion default values."""
        version = PolicyVersion(
            policy_id="p-1",
            version=1,
            policy_data={},
        )

        assert version.created_by is None
        assert version.change_description == ""
        assert isinstance(version.created_at, datetime)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPolicySystemIntegration:
    """Integration tests combining multiple policy system components."""

    @pytest.mark.asyncio
    async def test_full_policy_lifecycle_with_history(self):
        """Test complete policy lifecycle with history tracking."""
        manager = ControlPlanePolicyManager()
        history = PolicyHistory()

        # Create policy
        policy = ControlPlanePolicy(
            name="Lifecycle Policy",
            version=1,
            agent_allowlist=["claude-3-opus"],
        )
        manager.add_policy(policy)
        await history.record_version(policy, "Initial creation", "admin")

        # Update policy (simulated)
        updated_policy = ControlPlanePolicy(
            id=policy.id,
            name="Lifecycle Policy",
            version=2,
            agent_allowlist=["claude-3-opus", "gpt-4"],
            metadata={},
        )
        manager.remove_policy(policy.id)
        manager.add_policy(updated_policy)
        await history.record_version(updated_policy, "Added GPT-4", "admin")

        # Verify history
        versions = await history.get_history(policy.id)
        assert len(versions) == 2

        # Rollback
        restored = await history.rollback_to_version(policy.id, 1, "admin")
        assert restored is not None
        assert len(restored.agent_allowlist) == 1

    @pytest.mark.asyncio
    async def test_scheduler_with_cache_and_conflicts(self):
        """Test scheduler integrates cache and conflict detection."""
        manager = ControlPlanePolicyManager()
        mock_cache = AsyncMock(spec=RedisPolicyCache)
        conflicts_detected = []

        scheduler = PolicySyncScheduler(
            policy_manager=manager,
            sync_interval_seconds=1.0,
            policy_cache=mock_cache,
            conflict_callback=lambda c: conflicts_detected.extend(c),
            sync_from_compliance_store=False,
            sync_from_control_plane_store=False,
        )

        # Add conflicting policies
        manager.add_policy(
            ControlPlanePolicy(
                name="restrict",
                agent_allowlist=["agent-a"],
            )
        )
        manager.add_policy(
            ControlPlanePolicy(
                name="block",
                agent_blocklist=["agent-a"],
            )
        )

        # First sync - establish baseline
        await scheduler.sync_now()

        # Add more policies to trigger change
        manager.add_policy(ControlPlanePolicy(name="new-policy"))
        await scheduler.sync_now()

        # Verify
        assert len(conflicts_detected) > 0
        mock_cache.invalidate_all.assert_called()

    def test_manager_with_factory_policies(self):
        """Test manager with factory-created policies."""
        manager = ControlPlanePolicyManager()

        # Add various factory policies
        manager.add_policy(
            create_production_policy(
                agent_allowlist=["claude-3-opus"],
                allowed_regions=["us-east-1"],
            )
        )
        manager.add_policy(
            create_sla_policy(
                name="fast-sla",
                task_types=["fast-task"],
                max_execution_seconds=30.0,
            )
        )
        manager.add_policy(
            create_agent_tier_policy(
                tier="premium",
                agents=["claude-3-opus", "gpt-4"],
                task_types=["premium-task"],
            )
        )

        # Test evaluation
        result = manager.evaluate_task_dispatch(
            task_type="production-deployment",
            agent_id="claude-3-opus",
            region="us-east-1",
        )
        assert result.allowed is True

        result = manager.evaluate_task_dispatch(
            task_type="production-deployment",
            agent_id="gpt-3.5-turbo",
            region="us-east-1",
        )
        assert result.allowed is False
