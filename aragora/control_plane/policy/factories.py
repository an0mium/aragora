"""
Control Plane Policy Factories.

Pre-built policies for common scenarios.
"""

from __future__ import annotations

from typing import Optional

from .types import (
    ControlPlanePolicy,
    EnforcementLevel,
    RegionConstraint,
    SLARequirements,
)


# Pre-built policies for common scenarios
def create_production_policy(
    agent_allowlist: Optional[list[str]] = None,
    allowed_regions: Optional[list[str]] = None,
) -> ControlPlanePolicy:
    """Create a policy for production task restrictions."""
    return ControlPlanePolicy(
        name="production-restrictions",
        description="Restricts production tasks to approved agents and regions",
        task_types=["production-deployment", "production-migration"],
        agent_allowlist=agent_allowlist or [],
        region_constraint=RegionConstraint(
            allowed_regions=allowed_regions or [],
            require_data_residency=True,
        ),
        sla=SLARequirements(
            max_execution_seconds=600.0,
            max_queue_seconds=30.0,
            min_agents_available=2,
        ),
        enforcement_level=EnforcementLevel.HARD,
        priority=100,
    )


def create_sensitive_data_policy(
    data_regions: list[str],
    blocked_regions: Optional[list[str]] = None,
) -> ControlPlanePolicy:
    """Create a policy for sensitive data handling with residency requirements."""
    return ControlPlanePolicy(
        name="sensitive-data-residency",
        description="Enforces data residency for sensitive task types",
        task_types=["pii-processing", "financial-analysis", "healthcare-analysis"],
        region_constraint=RegionConstraint(
            allowed_regions=data_regions,
            blocked_regions=blocked_regions or [],
            require_data_residency=True,
            allow_cross_region=False,
        ),
        enforcement_level=EnforcementLevel.HARD,
        priority=90,
    )


def create_agent_tier_policy(
    tier: str,
    agents: list[str],
    task_types: Optional[list[str]] = None,
) -> ControlPlanePolicy:
    """Create a policy restricting certain task types to specific agent tiers."""
    return ControlPlanePolicy(
        name=f"{tier}-agent-tier",
        description=f"Restricts tasks to {tier} tier agents",
        task_types=task_types or [],
        agent_allowlist=agents,
        enforcement_level=EnforcementLevel.HARD,
        priority=50,
    )


def create_sla_policy(
    name: str,
    task_types: list[str],
    max_execution_seconds: float = 300.0,
    max_queue_seconds: float = 60.0,
) -> ControlPlanePolicy:
    """Create a policy with SLA enforcement."""
    return ControlPlanePolicy(
        name=name,
        description=f"SLA requirements for {', '.join(task_types)}",
        task_types=task_types,
        sla=SLARequirements(
            max_execution_seconds=max_execution_seconds,
            max_queue_seconds=max_queue_seconds,
        ),
        enforcement_level=EnforcementLevel.WARN,  # Warn on SLA violation
        priority=30,
    )
