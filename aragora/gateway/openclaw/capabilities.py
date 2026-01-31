"""
OpenClaw Capability Filter.

Provides enterprise policy enforcement for OpenClaw capabilities.
Capabilities are categorized into allow/block/require-approval buckets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CapabilityCategory(str, Enum):
    """Categories for capability access control."""

    ALWAYS_ALLOWED = "always_allowed"
    APPROVAL_REQUIRED = "approval_required"
    BLOCKED_BY_DEFAULT = "blocked_by_default"
    TENANT_CONFIGURABLE = "tenant_configurable"


@dataclass
class CapabilityRule:
    """Rule for a specific capability."""

    name: str
    category: CapabilityCategory
    description: str = ""
    approval_gate: str | None = None  # Required approval gate for APPROVAL_REQUIRED
    conditions: dict[str, Any] = field(default_factory=dict)


# Default capability rules
DEFAULT_CAPABILITY_RULES: dict[str, CapabilityRule] = {
    # Always allowed - low risk, core functionality
    "text_generation": CapabilityRule(
        name="text_generation",
        category=CapabilityCategory.ALWAYS_ALLOWED,
        description="Generate text responses",
    ),
    "search_internal": CapabilityRule(
        name="search_internal",
        category=CapabilityCategory.ALWAYS_ALLOWED,
        description="Search internal knowledge base",
    ),
    "calendar_read": CapabilityRule(
        name="calendar_read",
        category=CapabilityCategory.ALWAYS_ALLOWED,
        description="Read calendar events",
    ),
    "summarize": CapabilityRule(
        name="summarize",
        category=CapabilityCategory.ALWAYS_ALLOWED,
        description="Summarize text or documents",
    ),
    "translate": CapabilityRule(
        name="translate",
        category=CapabilityCategory.ALWAYS_ALLOWED,
        description="Translate text between languages",
    ),
    # Approval required - medium risk, sensitive operations
    "file_system_write": CapabilityRule(
        name="file_system_write",
        category=CapabilityCategory.APPROVAL_REQUIRED,
        description="Write files to filesystem",
        approval_gate="computer_use.file_write",
    ),
    "file_system_read": CapabilityRule(
        name="file_system_read",
        category=CapabilityCategory.APPROVAL_REQUIRED,
        description="Read files from filesystem",
        approval_gate="computer_use.file_read",
    ),
    "network_external": CapabilityRule(
        name="network_external",
        category=CapabilityCategory.APPROVAL_REQUIRED,
        description="Make external network requests",
        approval_gate="computer_use.network",
    ),
    "code_execution": CapabilityRule(
        name="code_execution",
        category=CapabilityCategory.APPROVAL_REQUIRED,
        description="Execute code in sandbox",
        approval_gate="computer_use.shell",
    ),
    "credential_access": CapabilityRule(
        name="credential_access",
        category=CapabilityCategory.APPROVAL_REQUIRED,
        description="Access stored credentials",
        approval_gate="api_key.read",
    ),
    "email_send": CapabilityRule(
        name="email_send",
        category=CapabilityCategory.APPROVAL_REQUIRED,
        description="Send emails on behalf of user",
        approval_gate="email.send",
    ),
    "calendar_write": CapabilityRule(
        name="calendar_write",
        category=CapabilityCategory.APPROVAL_REQUIRED,
        description="Create or modify calendar events",
        approval_gate="calendar.write",
    ),
    # Blocked by default - high risk, admin operations
    "shell_execute": CapabilityRule(
        name="shell_execute",
        category=CapabilityCategory.BLOCKED_BY_DEFAULT,
        description="Execute arbitrary shell commands",
    ),
    "admin_escalate": CapabilityRule(
        name="admin_escalate",
        category=CapabilityCategory.BLOCKED_BY_DEFAULT,
        description="Escalate to admin privileges",
    ),
    "data_export_bulk": CapabilityRule(
        name="data_export_bulk",
        category=CapabilityCategory.BLOCKED_BY_DEFAULT,
        description="Export large amounts of data",
    ),
    "system_config": CapabilityRule(
        name="system_config",
        category=CapabilityCategory.BLOCKED_BY_DEFAULT,
        description="Modify system configuration",
    ),
    "user_impersonate": CapabilityRule(
        name="user_impersonate",
        category=CapabilityCategory.BLOCKED_BY_DEFAULT,
        description="Act as another user",
    ),
    # Tenant configurable - can be enabled per-organization
    "browser_automation": CapabilityRule(
        name="browser_automation",
        category=CapabilityCategory.TENANT_CONFIGURABLE,
        description="Automate browser interactions",
    ),
    "database_query": CapabilityRule(
        name="database_query",
        category=CapabilityCategory.TENANT_CONFIGURABLE,
        description="Execute database queries",
    ),
    "api_integration": CapabilityRule(
        name="api_integration",
        category=CapabilityCategory.TENANT_CONFIGURABLE,
        description="Call third-party APIs",
    ),
}


@dataclass
class CapabilityCheckResult:
    """Result of a capability check."""

    allowed: bool
    capability: str
    category: CapabilityCategory
    reason: str
    requires_approval: bool = False
    approval_gate: str | None = None


class CapabilityFilter:
    """
    Filter OpenClaw capabilities based on enterprise policy.

    Enforces capability restrictions at the gateway level before
    tasks reach the OpenClaw runtime.
    """

    def __init__(
        self,
        rules: dict[str, CapabilityRule] | None = None,
        tenant_enabled: set[str] | None = None,
        blocked_override: set[str] | None = None,
    ) -> None:
        """
        Initialize capability filter.

        Args:
            rules: Custom capability rules (defaults to DEFAULT_CAPABILITY_RULES)
            tenant_enabled: Capabilities enabled for this tenant (for TENANT_CONFIGURABLE)
            blocked_override: Additional capabilities to block (on top of BLOCKED_BY_DEFAULT)
        """
        self.rules = rules or DEFAULT_CAPABILITY_RULES.copy()
        self.tenant_enabled = tenant_enabled or set()
        self.blocked_override = blocked_override or set()

    def check(
        self,
        capability: str,
        context: dict[str, Any] | None = None,
    ) -> CapabilityCheckResult:
        """
        Check if a capability is allowed.

        Args:
            capability: The capability being requested
            context: Additional context for conditional rules

        Returns:
            CapabilityCheckResult with the decision
        """
        # Check blocked override first
        if capability in self.blocked_override:
            return CapabilityCheckResult(
                allowed=False,
                capability=capability,
                category=CapabilityCategory.BLOCKED_BY_DEFAULT,
                reason=f"Capability '{capability}' is blocked by policy override",
            )

        # Get rule for capability
        rule = self.rules.get(capability)
        if rule is None:
            # Unknown capabilities are blocked by default
            return CapabilityCheckResult(
                allowed=False,
                capability=capability,
                category=CapabilityCategory.BLOCKED_BY_DEFAULT,
                reason=f"Unknown capability '{capability}' is blocked by default",
            )

        # Check by category
        if rule.category == CapabilityCategory.ALWAYS_ALLOWED:
            return CapabilityCheckResult(
                allowed=True,
                capability=capability,
                category=rule.category,
                reason="Capability is always allowed",
            )

        if rule.category == CapabilityCategory.BLOCKED_BY_DEFAULT:
            return CapabilityCheckResult(
                allowed=False,
                capability=capability,
                category=rule.category,
                reason=f"Capability '{capability}' is blocked by default",
            )

        if rule.category == CapabilityCategory.APPROVAL_REQUIRED:
            return CapabilityCheckResult(
                allowed=False,  # Not immediately allowed
                capability=capability,
                category=rule.category,
                reason=f"Capability '{capability}' requires approval",
                requires_approval=True,
                approval_gate=rule.approval_gate,
            )

        if rule.category == CapabilityCategory.TENANT_CONFIGURABLE:
            if capability in self.tenant_enabled:
                return CapabilityCheckResult(
                    allowed=True,
                    capability=capability,
                    category=rule.category,
                    reason="Capability is enabled for this tenant",
                )
            return CapabilityCheckResult(
                allowed=False,
                capability=capability,
                category=rule.category,
                reason=f"Capability '{capability}' is not enabled for this tenant",
            )

        # Fallback - block unknown categories
        return CapabilityCheckResult(
            allowed=False,
            capability=capability,
            category=CapabilityCategory.BLOCKED_BY_DEFAULT,
            reason="Unknown capability category",
        )

    def check_multiple(
        self,
        capabilities: list[str],
        context: dict[str, Any] | None = None,
    ) -> dict[str, CapabilityCheckResult]:
        """
        Check multiple capabilities at once.

        Args:
            capabilities: List of capabilities to check
            context: Additional context for conditional rules

        Returns:
            Dict mapping capability name to check result
        """
        return {cap: self.check(cap, context) for cap in capabilities}

    def get_blocked_capabilities(
        self,
        capabilities: list[str],
    ) -> list[str]:
        """Get list of capabilities that would be blocked."""
        return [cap for cap in capabilities if not self.check(cap).allowed]

    def enable_for_tenant(self, capability: str) -> None:
        """Enable a tenant-configurable capability."""
        rule = self.rules.get(capability)
        if rule and rule.category == CapabilityCategory.TENANT_CONFIGURABLE:
            self.tenant_enabled.add(capability)

    def disable_for_tenant(self, capability: str) -> None:
        """Disable a tenant-configurable capability."""
        self.tenant_enabled.discard(capability)

    def add_block_override(self, capability: str) -> None:
        """Add a capability to the blocked override list."""
        self.blocked_override.add(capability)

    def remove_block_override(self, capability: str) -> None:
        """Remove a capability from the blocked override list."""
        self.blocked_override.discard(capability)


__all__ = [
    "CapabilityCategory",
    "CapabilityRule",
    "CapabilityCheckResult",
    "CapabilityFilter",
    "DEFAULT_CAPABILITY_RULES",
]
