"""
Control Plane Policy Factories.

Pre-built policies for common scenarios.
"""

from __future__ import annotations


from .types import (
    ControlPlanePolicy,
    EnforcementLevel,
    RegionConstraint,
    SLARequirements,
)


# Pre-built policies for common scenarios
def create_production_policy(
    name: str = "production-environment",
    allowed_agents: list[str] | None = None,
    blocked_agents: list[str] | None = None,
    allowed_regions: list[str] | None = None,
    # Aliases for backward compatibility
    agent_allowlist: list[str] | None = None,
) -> ControlPlanePolicy:
    """Create a policy for production task restrictions.

    Args:
        name: Policy name (default: "production-environment")
        allowed_agents: List of agent IDs that are allowed
        blocked_agents: List of agent IDs that are blocked
        allowed_regions: List of allowed regions for deployment
        agent_allowlist: Alias for allowed_agents (deprecated)
    """
    # Support alias
    effective_agents = allowed_agents or agent_allowlist or []
    return ControlPlanePolicy(
        name=name,
        description="Restricts production tasks to approved agents and regions",
        task_types=["production", "production-deployment", "production-migration"],
        agent_allowlist=effective_agents,
        agent_blocklist=blocked_agents or [],
        region_constraint=RegionConstraint(
            allowed_regions=allowed_regions or [],
            require_data_residency=True if allowed_regions else False,
        ),
        sla=SLARequirements(
            max_execution_seconds=600.0,
            max_queue_seconds=30.0,
            min_agents_available=2,
        ),
        enforcement_level=EnforcementLevel.HARD,
        priority=100,
        enabled=True,
    )


def create_sensitive_data_policy(
    allowed_agents: list[str] | None = None,
    allowed_regions: list[str] | None = None,
    blocked_regions: list[str] | None = None,
    require_data_residency: bool = True,
    # Alias for backward compatibility
    data_regions: list[str] | None = None,
) -> ControlPlanePolicy:
    """Create a policy for sensitive data handling with residency requirements.

    Args:
        allowed_agents: List of agent IDs allowed to process sensitive data
        allowed_regions: List of regions where data can be processed
        blocked_regions: List of regions where data cannot be processed
        require_data_residency: Whether to enforce data residency requirements
        data_regions: Alias for allowed_regions (deprecated)
    """
    # Support alias
    effective_regions = allowed_regions or data_regions or []
    return ControlPlanePolicy(
        name="sensitive-data-residency",
        description="Enforces data residency for sensitive task types",
        task_types=["pii-processing", "financial-analysis", "healthcare-analysis"],
        agent_allowlist=allowed_agents or [],
        region_constraint=RegionConstraint(
            allowed_regions=effective_regions,
            blocked_regions=blocked_regions or [],
            require_data_residency=require_data_residency,
            allow_cross_region=False,
        ),
        enforcement_level=EnforcementLevel.HARD,
        priority=90,
        enabled=True,
    )


def create_agent_tier_policy(
    tier: str,
    allowed_agents: list[str] | None = None,
    task_types: list[str] | None = None,
    # Alias for backward compatibility
    agents: list[str] | None = None,
) -> ControlPlanePolicy:
    """Create a policy restricting certain task types to specific agent tiers.

    Args:
        tier: The tier name (e.g., "premium", "standard")
        allowed_agents: List of agent IDs in this tier
        task_types: Task types that require this tier (optional)
        agents: Alias for allowed_agents (deprecated)
    """
    # Support alias
    effective_agents = allowed_agents or agents or []
    return ControlPlanePolicy(
        name=f"{tier}-agent-tier",
        description=f"Restricts tasks to {tier} tier agents",
        task_types=task_types or [],
        agent_allowlist=effective_agents,
        enforcement_level=EnforcementLevel.HARD,
        priority=50,
        enabled=True,
    )


def create_sla_policy(
    name: str = "default-sla",
    task_types: list[str] | None = None,
    max_execution_seconds: float = 300.0,
    max_queue_seconds: float = 60.0,
    max_concurrent_tasks: int = 5,
    enforcement_level: EnforcementLevel = EnforcementLevel.WARN,
) -> ControlPlanePolicy:
    """Create a policy with SLA enforcement.

    Args:
        name: Policy name
        task_types: Task types this SLA applies to
        max_execution_seconds: Maximum execution time
        max_queue_seconds: Maximum queue wait time
        max_concurrent_tasks: Maximum concurrent tasks per agent
        enforcement_level: How strictly to enforce (default: WARN)
    """
    effective_task_types = task_types or ["default"]
    return ControlPlanePolicy(
        name=name,
        description=f"SLA requirements for {', '.join(effective_task_types)}",
        task_types=effective_task_types,
        sla=SLARequirements(
            max_execution_seconds=max_execution_seconds,
            max_queue_seconds=max_queue_seconds,
            max_concurrent_tasks=max_concurrent_tasks,
        ),
        enforcement_level=enforcement_level,
        priority=30,
        enabled=True,
    )
