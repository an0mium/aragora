"""
Tool Policy Definitions for Sandbox Isolation.

Defines what tools and operations are allowed within sandboxed execution:
- Allowlist/denylist based filtering
- Resource access controls
- Network restrictions
- File system boundaries
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PolicyAction(str, Enum):
    """Actions that can be taken on a tool request."""

    ALLOW = "allow"
    DENY = "deny"
    AUDIT = "audit"  # Allow but log


class ResourceType(str, Enum):
    """Types of resources that can be controlled."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    NETWORK = "network"
    PROCESS = "process"
    ENVIRONMENT = "environment"
    MEMORY = "memory"
    CPU = "cpu"


@dataclass
class ToolRule:
    """A rule for tool access."""

    pattern: str  # Regex pattern for tool name
    action: PolicyAction = PolicyAction.ALLOW
    reason: str = ""
    resource_limits: dict[str, Any] = field(default_factory=dict)

    def matches(self, tool_name: str) -> bool:
        """Check if this rule matches a tool name."""
        return bool(re.match(self.pattern, tool_name))


@dataclass
class PathRule:
    """A rule for file path access."""

    pattern: str  # Glob or regex pattern
    action: PolicyAction = PolicyAction.ALLOW
    read_allowed: bool = True
    write_allowed: bool = False
    reason: str = ""

    def matches(self, path: str) -> bool:
        """Check if this rule matches a path."""
        return bool(re.match(self.pattern, path))


@dataclass
class NetworkRule:
    """A rule for network access."""

    host_pattern: str  # Regex for host/domain
    port_range: tuple[int, int] = (1, 65535)
    action: PolicyAction = PolicyAction.ALLOW
    protocols: list[str] = field(default_factory=lambda: ["http", "https"])
    reason: str = ""

    def matches(self, host: str, port: int, protocol: str) -> bool:
        """Check if this rule matches a network request."""
        if not re.match(self.host_pattern, host):
            return False
        if not (self.port_range[0] <= port <= self.port_range[1]):
            return False
        if protocol not in self.protocols:
            return False
        return True


@dataclass
class ResourceLimit:
    """Resource limits for sandboxed execution."""

    max_memory_mb: int = 512
    max_cpu_percent: int = 100
    max_execution_seconds: int = 60
    max_processes: int = 10
    max_file_size_mb: int = 10
    max_files_created: int = 100
    max_network_requests: int = 50


@dataclass
class ToolPolicy:
    """
    Complete policy definition for sandboxed execution.

    Combines tool rules, path rules, network rules, and resource limits
    to define what a sandboxed process can do.
    """

    name: str
    description: str = ""

    # Default actions
    default_tool_action: PolicyAction = PolicyAction.DENY
    default_path_action: PolicyAction = PolicyAction.DENY
    default_network_action: PolicyAction = PolicyAction.DENY

    # Rules
    tool_rules: list[ToolRule] = field(default_factory=list)
    path_rules: list[PathRule] = field(default_factory=list)
    network_rules: list[NetworkRule] = field(default_factory=list)

    # Resource limits
    resource_limits: ResourceLimit = field(default_factory=ResourceLimit)

    # Audit settings
    audit_all: bool = False
    audit_denials: bool = True

    def add_tool_allowlist(self, patterns: list[str], reason: str = "") -> None:
        """Add tools to the allowlist."""
        for pattern in patterns:
            self.tool_rules.append(
                ToolRule(pattern=pattern, action=PolicyAction.ALLOW, reason=reason)
            )

    def add_tool_denylist(self, patterns: list[str], reason: str = "") -> None:
        """Add tools to the denylist."""
        for pattern in patterns:
            self.tool_rules.append(
                ToolRule(pattern=pattern, action=PolicyAction.DENY, reason=reason)
            )

    def add_path_allowlist(
        self,
        patterns: list[str],
        read: bool = True,
        write: bool = False,
        reason: str = "",
    ) -> None:
        """Add paths to the allowlist."""
        for pattern in patterns:
            self.path_rules.append(
                PathRule(
                    pattern=pattern,
                    action=PolicyAction.ALLOW,
                    read_allowed=read,
                    write_allowed=write,
                    reason=reason,
                )
            )

    def add_network_allowlist(
        self,
        host_patterns: list[str],
        ports: tuple[int, int] = (80, 443),
        reason: str = "",
    ) -> None:
        """Add network hosts to the allowlist."""
        for pattern in host_patterns:
            self.network_rules.append(
                NetworkRule(
                    host_pattern=pattern,
                    port_range=ports,
                    action=PolicyAction.ALLOW,
                    reason=reason,
                )
            )


class ToolPolicyChecker:
    """
    Checks tool/resource requests against a policy.

    Provides consistent policy enforcement with audit logging.
    """

    def __init__(self, policy: ToolPolicy):
        self.policy = policy
        self._audit_log: list[dict[str, Any]] = []

    def check_tool(self, tool_name: str) -> tuple[bool, str]:
        """
        Check if a tool is allowed.

        Returns:
            (allowed, reason) tuple
        """
        # Check rules in order (first match wins)
        for rule in self.policy.tool_rules:
            if rule.matches(tool_name):
                allowed = rule.action in (PolicyAction.ALLOW, PolicyAction.AUDIT)
                self._log_check("tool", tool_name, allowed, rule.reason)
                return allowed, rule.reason

        # Fall back to default
        allowed = self.policy.default_tool_action in (PolicyAction.ALLOW, PolicyAction.AUDIT)
        self._log_check("tool", tool_name, allowed, "default policy")
        return allowed, "default policy"

    def check_path(
        self,
        path: str,
        operation: str = "read",
    ) -> tuple[bool, str]:
        """
        Check if a file path access is allowed.

        Args:
            path: File path
            operation: "read" or "write"

        Returns:
            (allowed, reason) tuple
        """
        for rule in self.policy.path_rules:
            if rule.matches(path):
                if operation == "read":
                    allowed = rule.read_allowed and rule.action != PolicyAction.DENY
                else:
                    allowed = rule.write_allowed and rule.action != PolicyAction.DENY
                self._log_check(f"path_{operation}", path, allowed, rule.reason)
                return allowed, rule.reason

        allowed = self.policy.default_path_action in (PolicyAction.ALLOW, PolicyAction.AUDIT)
        self._log_check(f"path_{operation}", path, allowed, "default policy")
        return allowed, "default policy"

    def check_network(
        self,
        host: str,
        port: int = 443,
        protocol: str = "https",
    ) -> tuple[bool, str]:
        """
        Check if a network request is allowed.

        Returns:
            (allowed, reason) tuple
        """
        for rule in self.policy.network_rules:
            if rule.matches(host, port, protocol):
                allowed = rule.action in (PolicyAction.ALLOW, PolicyAction.AUDIT)
                self._log_check("network", f"{protocol}://{host}:{port}", allowed, rule.reason)
                return allowed, rule.reason

        allowed = self.policy.default_network_action in (PolicyAction.ALLOW, PolicyAction.AUDIT)
        self._log_check("network", f"{protocol}://{host}:{port}", allowed, "default policy")
        return allowed, "default policy"

    def get_resource_limits(self) -> ResourceLimit:
        """Get the resource limits from the policy."""
        return self.policy.resource_limits

    def _log_check(
        self,
        check_type: str,
        resource: str,
        allowed: bool,
        reason: str,
    ) -> None:
        """Log a policy check."""
        entry = {
            "type": check_type,
            "resource": resource,
            "allowed": allowed,
            "reason": reason,
        }

        if self.policy.audit_all or (not allowed and self.policy.audit_denials):
            logger.info(f"Policy check: {entry}")

        self._audit_log.append(entry)

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get the audit log."""
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()


def create_default_policy() -> ToolPolicy:
    """Create a sensible default policy for agent sandbox execution."""
    policy = ToolPolicy(
        name="default",
        description="Default sandbox policy for safe agent execution",
        default_tool_action=PolicyAction.DENY,
        default_path_action=PolicyAction.DENY,
        default_network_action=PolicyAction.DENY,
    )

    # Allow common safe tools
    policy.add_tool_allowlist(
        [
            r"^python$",
            r"^python3$",
            r"^pip$",
            r"^pip3$",
            r"^node$",
            r"^npm$",
            r"^cat$",
            r"^head$",
            r"^tail$",
            r"^grep$",
            r"^find$",
            r"^ls$",
            r"^echo$",
            r"^printf$",
            r"^wc$",
            r"^sort$",
            r"^uniq$",
            r"^jq$",
        ],
        reason="Safe standard tools",
    )

    # Deny dangerous tools
    policy.add_tool_denylist(
        [
            r"^rm$",
            r"^rmdir$",
            r"^dd$",
            r"^mkfs.*$",
            r"^fdisk$",
            r"^format$",
            r"^curl$",
            r"^wget$",
            r"^nc$",
            r"^netcat$",
            r"^ssh$",
            r"^scp$",
            r"^rsync$",
            r"^chmod$",
            r"^chown$",
            r"^sudo$",
            r"^su$",
            r"^kill$",
            r"^pkill$",
            r"^killall$",
        ],
        reason="Dangerous system tools",
    )

    # Allow read access to safe paths
    policy.add_path_allowlist(
        [
            r"^/tmp/sandbox/.*$",
            r"^/workspace/.*$",
            r"^\./.*$",
        ],
        read=True,
        write=True,
        reason="Sandbox workspace",
    )

    # Allow read-only access to system libraries
    policy.add_path_allowlist(
        [
            r"^/usr/lib/.*$",
            r"^/usr/share/.*$",
            r"^/etc/localtime$",
        ],
        read=True,
        write=False,
        reason="System libraries (read-only)",
    )

    # Allow localhost network for testing
    policy.add_network_allowlist(
        [r"^localhost$", r"^127\.0\.0\.1$"],
        ports=(1024, 65535),
        reason="Localhost for testing",
    )

    return policy


def create_strict_policy() -> ToolPolicy:
    """Create a strict policy with minimal permissions."""
    policy = ToolPolicy(
        name="strict",
        description="Strict sandbox policy with minimal permissions",
        default_tool_action=PolicyAction.DENY,
        default_path_action=PolicyAction.DENY,
        default_network_action=PolicyAction.DENY,
        audit_all=True,
    )

    # Only allow basic read tools
    policy.add_tool_allowlist(
        [r"^cat$", r"^head$", r"^tail$", r"^ls$", r"^echo$"],
        reason="Basic read-only tools",
    )

    # Only allow sandbox directory
    policy.add_path_allowlist(
        [r"^/tmp/sandbox/.*$"],
        read=True,
        write=True,
        reason="Sandbox directory only",
    )

    # Very limited resources
    policy.resource_limits = ResourceLimit(
        max_memory_mb=256,
        max_cpu_percent=50,
        max_execution_seconds=30,
        max_processes=5,
        max_file_size_mb=5,
        max_files_created=10,
        max_network_requests=0,
    )

    return policy


def create_permissive_policy() -> ToolPolicy:
    """Create a permissive policy for trusted environments."""
    policy = ToolPolicy(
        name="permissive",
        description="Permissive policy for trusted agent execution",
        default_tool_action=PolicyAction.ALLOW,
        default_path_action=PolicyAction.ALLOW,
        default_network_action=PolicyAction.AUDIT,
    )

    # Still deny the most dangerous tools
    policy.add_tool_denylist(
        [r"^rm\s+-rf\s+/$", r"^dd\s+.*of=/dev/.*$", r"^:(){ :|:& };:$"],
        reason="Extremely dangerous operations",
    )

    # More generous limits
    policy.resource_limits = ResourceLimit(
        max_memory_mb=2048,
        max_cpu_percent=200,
        max_execution_seconds=300,
        max_processes=50,
        max_file_size_mb=100,
        max_files_created=1000,
        max_network_requests=500,
    )

    return policy


__all__ = [
    "NetworkRule",
    "PathRule",
    "PolicyAction",
    "ResourceLimit",
    "ResourceType",
    "ToolPolicy",
    "ToolPolicyChecker",
    "ToolRule",
    "create_default_policy",
    "create_permissive_policy",
    "create_strict_policy",
]
