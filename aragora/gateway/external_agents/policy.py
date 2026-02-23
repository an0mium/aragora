"""
External Agent Policy Enforcement.

Defines policies for controlling external agent behavior including:
- Capability allow/block/require-approval rules
- Sensitive data classification routing
- Tenant-specific policy overrides
- Emergency policy escalation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.gateway.external_agents.base import (
        ExternalAgentAdapter,
        ExternalAgentTask,
    )

logger = logging.getLogger(__name__)


class PolicyAction(str, Enum):
    """Actions that can be taken on a capability request."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    AUDIT_ONLY = "audit_only"  # Allow but log for review


class SensitivityLevel(str, Enum):
    """Data sensitivity classification."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class PolicyDecision:
    """Result of a policy evaluation."""

    allowed: bool
    action: PolicyAction
    reason: str = ""
    requires_approval: bool = False
    approval_id: str | None = None
    audit_required: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass
class PolicyViolation:
    """Details of a policy violation."""

    policy_id: str
    policy_name: str
    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    details: str
    remediation: str | None = None


@dataclass
class CapabilityRule:
    """Rule for a specific capability."""

    capability: str  # AgentCapability value
    action: PolicyAction = PolicyAction.ALLOW
    conditions: dict[str, Any] = field(default_factory=dict)
    # Conditions can include:
    # - tenant_ids: list[str] - Only apply for these tenants
    # - exclude_tenant_ids: list[str] - Don't apply for these tenants
    # - sensitivity_levels: list[str] - Apply based on data sensitivity
    # - time_window: dict - Only during certain times
    # - rate_limit: dict - Rate limiting config


@dataclass
class ExternalAgentPolicy:
    """
    Policy configuration for external agent execution.

    Defines what capabilities external agents can use and under what
    conditions, providing fine-grained control over agent behavior.
    """

    policy_id: str
    policy_name: str
    description: str = ""

    # Global settings
    enabled: bool = True
    default_action: PolicyAction = PolicyAction.DENY

    # Capability rules
    capability_rules: list[CapabilityRule] = field(default_factory=list)

    # Blocked agents (by name)
    blocked_agents: list[str] = field(default_factory=list)

    # Allowed agents (if set, only these are allowed)
    allowed_agents: list[str] | None = None

    # Sensitivity routing - route sensitive tasks to aragora agents only
    sensitivity_threshold: SensitivityLevel = SensitivityLevel.CONFIDENTIAL

    # Tenant overrides
    tenant_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Rate limiting
    max_executions_per_minute: int = 60
    max_executions_per_hour: int = 1000
    max_tokens_per_hour: int = 1_000_000

    def get_capability_action(
        self,
        capability: str,
        tenant_id: str | None = None,
        sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL,
    ) -> PolicyAction:
        """Get the policy action for a capability.

        Args:
            capability: The capability to check
            tenant_id: Optional tenant context
            sensitivity: Data sensitivity level

        Returns:
            PolicyAction indicating how to handle the capability
        """
        # Check tenant overrides first
        if tenant_id and tenant_id in self.tenant_overrides:
            override = self.tenant_overrides[tenant_id]
            if "blocked_capabilities" in override:
                if capability in override["blocked_capabilities"]:
                    return PolicyAction.DENY
            if "allowed_capabilities" in override:
                if capability in override["allowed_capabilities"]:
                    return PolicyAction.ALLOW

        # Check specific rules
        for rule in self.capability_rules:
            if rule.capability == capability:
                # Check conditions
                if self._rule_applies(rule, tenant_id, sensitivity):
                    return rule.action

        return self.default_action

    def _rule_applies(
        self,
        rule: CapabilityRule,
        tenant_id: str | None,
        sensitivity: SensitivityLevel,
    ) -> bool:
        """Check if a rule applies given the context."""
        conditions = rule.conditions

        # Check tenant conditions
        if "tenant_ids" in conditions:
            if tenant_id not in conditions["tenant_ids"]:
                return False
        if "exclude_tenant_ids" in conditions:
            if tenant_id in conditions["exclude_tenant_ids"]:
                return False

        # Check sensitivity conditions
        if "sensitivity_levels" in conditions:
            if sensitivity.value not in conditions["sensitivity_levels"]:
                return False

        return True


class PolicyEngine:
    """
    Engine for evaluating policies against external agent tasks.

    Provides centralized policy enforcement with support for:
    - Multiple policy layers (global, tenant, user)
    - Policy caching for performance
    - Audit logging of all decisions
    """

    def __init__(
        self,
        default_policy: ExternalAgentPolicy | None = None,
        audit_logger: Any | None = None,
    ):
        self._default_policy = default_policy or self._create_default_policy()
        self._tenant_policies: dict[str, ExternalAgentPolicy] = {}
        self._audit_logger = audit_logger

    def _create_default_policy(self) -> ExternalAgentPolicy:
        """Create the default enterprise security policy."""
        from aragora.gateway.external_agents.base import AgentCapability

        return ExternalAgentPolicy(
            policy_id="default-enterprise",
            policy_name="Default Enterprise Policy",
            description="Secure defaults for enterprise use",
            default_action=PolicyAction.DENY,
            capability_rules=[
                # Allow safe capabilities
                CapabilityRule(
                    capability=AgentCapability.WEB_SEARCH.value,
                    action=PolicyAction.ALLOW,
                ),
                CapabilityRule(
                    capability=AgentCapability.WEB_BROWSE.value,
                    action=PolicyAction.AUDIT_ONLY,
                ),
                # Require approval for file access
                CapabilityRule(
                    capability=AgentCapability.FILE_READ.value,
                    action=PolicyAction.REQUIRE_APPROVAL,
                ),
                CapabilityRule(
                    capability=AgentCapability.FILE_WRITE.value,
                    action=PolicyAction.REQUIRE_APPROVAL,
                ),
                # Block dangerous capabilities
                CapabilityRule(
                    capability=AgentCapability.SHELL_ACCESS.value,
                    action=PolicyAction.DENY,
                ),
                CapabilityRule(
                    capability=AgentCapability.EXECUTE_CODE.value,
                    action=PolicyAction.DENY,
                ),
                CapabilityRule(
                    capability=AgentCapability.SCREEN_CAPTURE.value,
                    action=PolicyAction.DENY,
                ),
            ],
            sensitivity_threshold=SensitivityLevel.CONFIDENTIAL,
        )

    def set_tenant_policy(self, tenant_id: str, policy: ExternalAgentPolicy) -> None:
        """Set a tenant-specific policy."""
        self._tenant_policies[tenant_id] = policy
        logger.info("Set policy for tenant %s: %s", tenant_id, policy.policy_name)

    def get_policy(self, tenant_id: str | None = None) -> ExternalAgentPolicy:
        """Get the applicable policy for a tenant."""
        if tenant_id and tenant_id in self._tenant_policies:
            return self._tenant_policies[tenant_id]
        return self._default_policy

    async def evaluate(
        self,
        adapter: ExternalAgentAdapter,
        task: ExternalAgentTask,
        tenant_id: str | None = None,
        user_id: str | None = None,
        sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL,
    ) -> PolicyDecision:
        """
        Evaluate a task against the applicable policy.

        Args:
            adapter: The agent adapter to be used
            task: The task to be executed
            tenant_id: Tenant context
            user_id: User context
            sensitivity: Data sensitivity level

        Returns:
            PolicyDecision with the evaluation result
        """
        policy = self.get_policy(tenant_id)
        violations: list[str] = []
        warnings: list[str] = []
        requires_approval = False

        # Check if policy is enabled
        if not policy.enabled:
            return PolicyDecision(
                allowed=True,
                action=PolicyAction.ALLOW,
                reason="Policy disabled",
            )

        # Check if agent is blocked
        if adapter.agent_name in policy.blocked_agents:
            return PolicyDecision(
                allowed=False,
                action=PolicyAction.DENY,
                reason=f"Agent {adapter.agent_name} is blocked by policy",
            )

        # Check if agent is in allowed list (if set)
        if policy.allowed_agents is not None:
            if adapter.agent_name not in policy.allowed_agents:
                return PolicyDecision(
                    allowed=False,
                    action=PolicyAction.DENY,
                    reason=f"Agent {adapter.agent_name} is not in allowed list",
                )

        # Check sensitivity routing
        sensitivity_order = [
            SensitivityLevel.PUBLIC,
            SensitivityLevel.INTERNAL,
            SensitivityLevel.CONFIDENTIAL,
            SensitivityLevel.RESTRICTED,
        ]
        if sensitivity_order.index(sensitivity) >= sensitivity_order.index(
            policy.sensitivity_threshold
        ):
            return PolicyDecision(
                allowed=False,
                action=PolicyAction.DENY,
                reason=(
                    f"Data sensitivity {sensitivity.value} exceeds threshold "
                    f"{policy.sensitivity_threshold.value} - use aragora agents only"
                ),
            )

        # Check each required capability
        for capability in task.required_capabilities:
            action = policy.get_capability_action(capability.value, tenant_id, sensitivity)

            if action == PolicyAction.DENY:
                violations.append(f"Capability {capability.value} is denied")
            elif action == PolicyAction.REQUIRE_APPROVAL:
                requires_approval = True
                warnings.append(f"Capability {capability.value} requires approval")
            elif action == PolicyAction.AUDIT_ONLY:
                warnings.append(f"Capability {capability.value} will be audited")

        # If any violations, deny
        if violations:
            decision = PolicyDecision(
                allowed=False,
                action=PolicyAction.DENY,
                reason="; ".join(violations),
                warnings=warnings,
            )
        elif requires_approval:
            decision = PolicyDecision(
                allowed=False,
                action=PolicyAction.REQUIRE_APPROVAL,
                reason="Approval required for requested capabilities",
                requires_approval=True,
                warnings=warnings,
            )
        else:
            decision = PolicyDecision(
                allowed=True,
                action=PolicyAction.ALLOW,
                reason="All checks passed",
                warnings=warnings,
            )

        # Log decision
        if self._audit_logger:
            await self._audit_logger.log_policy_decision(
                policy_id=policy.policy_id,
                adapter_name=adapter.agent_name,
                task_id=task.task_id,
                tenant_id=tenant_id,
                user_id=user_id,
                decision=decision,
            )

        return decision
