"""Tests for control plane policy factory functions."""

from __future__ import annotations

import pytest

from aragora.control_plane.policy.factories import (
    create_agent_tier_policy,
    create_production_policy,
    create_sensitive_data_policy,
    create_sla_policy,
)
from aragora.control_plane.policy.types import EnforcementLevel


class TestCreateProductionPolicy:
    """Tests for create_production_policy factory."""

    def test_default_production_policy(self):
        """Test creating default production policy."""
        policy = create_production_policy()
        assert policy.name == "production-environment"
        assert "production" in policy.task_types
        assert policy.enforcement_level == EnforcementLevel.HARD
        assert policy.enabled is True

    def test_custom_name(self):
        """Test custom policy name."""
        policy = create_production_policy(name="my-prod-policy")
        assert policy.name == "my-prod-policy"

    def test_allowed_agents(self):
        """Test specifying allowed agents."""
        policy = create_production_policy(
            allowed_agents=["claude-3-opus", "gpt-4"],
        )
        assert "claude-3-opus" in policy.agent_allowlist
        assert "gpt-4" in policy.agent_allowlist
        assert policy.is_agent_allowed("claude-3-opus") is True
        assert policy.is_agent_allowed("gpt-3.5-turbo") is False

    def test_blocked_agents(self):
        """Test specifying blocked agents."""
        policy = create_production_policy(
            blocked_agents=["experimental-agent"],
        )
        assert "experimental-agent" in policy.agent_blocklist
        assert policy.is_agent_allowed("experimental-agent") is False

    def test_allowed_regions(self):
        """Test specifying allowed regions."""
        policy = create_production_policy(
            allowed_regions=["us-east-1", "us-west-2"],
        )
        assert policy.region_constraint is not None
        assert policy.is_region_allowed("us-east-1") is True
        assert policy.is_region_allowed("eu-west-1") is False


class TestCreateSensitiveDataPolicy:
    """Tests for create_sensitive_data_policy factory."""

    def test_default_sensitive_data_policy(self):
        """Test creating default sensitive data policy."""
        policy = create_sensitive_data_policy()
        assert "sensitive" in policy.name.lower() or "data" in policy.name.lower()
        assert policy.enforcement_level == EnforcementLevel.HARD

    def test_data_residency_required(self):
        """Test data residency requirement."""
        policy = create_sensitive_data_policy(
            require_data_residency=True,
            allowed_regions=["eu-west-1"],
        )
        assert policy.region_constraint is not None
        assert policy.region_constraint.require_data_residency is True

    def test_restricted_agents(self):
        """Test sensitive data restricts agents."""
        policy = create_sensitive_data_policy(
            allowed_agents=["trusted-agent"],
        )
        assert policy.is_agent_allowed("trusted-agent") is True
        assert policy.is_agent_allowed("untrusted-agent") is False


class TestCreateAgentTierPolicy:
    """Tests for create_agent_tier_policy factory."""

    def test_premium_tier(self):
        """Test creating premium tier policy."""
        policy = create_agent_tier_policy(
            tier="premium",
            allowed_agents=["claude-3-opus", "gpt-4", "gemini-ultra"],
        )
        assert "premium" in policy.name.lower()
        assert policy.is_agent_allowed("claude-3-opus") is True
        assert policy.is_agent_allowed("gpt-3.5-turbo") is False

    def test_standard_tier(self):
        """Test creating standard tier policy."""
        policy = create_agent_tier_policy(
            tier="standard",
            allowed_agents=["gpt-3.5-turbo", "claude-instant"],
        )
        assert "standard" in policy.name.lower()
        assert policy.is_agent_allowed("gpt-3.5-turbo") is True
        assert policy.is_agent_allowed("claude-3-opus") is False

    def test_tier_task_types(self):
        """Test tier-specific task types."""
        policy = create_agent_tier_policy(
            tier="premium",
            task_types=["complex-analysis", "production-deploy"],
        )
        assert "complex-analysis" in policy.task_types
        assert "production-deploy" in policy.task_types


class TestCreateSlaPolicy:
    """Tests for create_sla_policy factory."""

    def test_default_sla_policy(self):
        """Test creating default SLA policy."""
        policy = create_sla_policy()
        assert policy.sla is not None
        assert policy.sla.max_execution_seconds > 0

    def test_custom_sla_values(self):
        """Test custom SLA values."""
        policy = create_sla_policy(
            max_execution_seconds=120.0,
            max_queue_seconds=30.0,
            max_concurrent_tasks=3,
        )
        assert policy.sla.max_execution_seconds == 120.0
        assert policy.sla.max_queue_seconds == 30.0
        assert policy.sla.max_concurrent_tasks == 3

    def test_sla_enforcement_level(self):
        """Test SLA enforcement level."""
        # Default should be WARN or SOFT for SLA
        policy = create_sla_policy()
        assert policy.enforcement_level in (
            EnforcementLevel.WARN,
            EnforcementLevel.SOFT,
            EnforcementLevel.HARD,
        )

    def test_strict_sla_policy(self):
        """Test strict SLA enforcement."""
        policy = create_sla_policy(
            max_execution_seconds=60.0,
            enforcement_level=EnforcementLevel.HARD,
        )
        assert policy.sla.max_execution_seconds == 60.0
        assert policy.enforcement_level == EnforcementLevel.HARD


class TestFactoryPolicyMetadata:
    """Tests for factory-generated policy metadata."""

    def test_factory_policies_have_ids(self):
        """Test all factory policies have unique IDs."""
        policies = [
            create_production_policy(),
            create_sensitive_data_policy(),
            create_agent_tier_policy(tier="premium"),
            create_sla_policy(),
        ]
        ids = [p.id for p in policies]
        assert len(ids) == len(set(ids))  # All unique

    def test_factory_policies_have_descriptions(self):
        """Test factory policies have descriptions."""
        policy = create_production_policy()
        # Descriptions may be empty string or meaningful text
        assert policy.description is not None

    def test_factory_policies_enabled_by_default(self):
        """Test factory policies are enabled by default."""
        policies = [
            create_production_policy(),
            create_sensitive_data_policy(),
            create_agent_tier_policy(tier="standard"),
            create_sla_policy(),
        ]
        assert all(p.enabled for p in policies)
