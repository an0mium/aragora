"""
Policies API resource for the Aragora client.

Provides methods for compliance policy management:
- Policy CRUD operations
- Policy violation tracking
- Compliance checks and statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class PolicyRule:
    """A rule within a policy."""

    id: str
    name: str
    description: str
    condition: str
    severity: str  # critical, high, medium, low
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Policy:
    """A compliance policy."""

    id: str
    name: str
    description: str
    framework_id: str
    workspace_id: str
    vertical_id: str
    level: str  # required, recommended, optional
    enabled: bool = True
    rules: List[PolicyRule] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyViolation:
    """A policy violation."""

    id: str
    policy_id: str
    rule_id: str
    rule_name: str
    framework_id: str
    vertical_id: str
    workspace_id: str
    severity: str
    status: str  # open, investigating, resolved, false_positive
    description: str
    source: str
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceCheckResult:
    """Result of a compliance check."""

    compliant: bool
    score: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    checked_at: Optional[datetime] = None


@dataclass
class ComplianceStats:
    """Compliance statistics."""

    policies_total: int = 0
    policies_enabled: int = 0
    policies_disabled: int = 0
    violations_total: int = 0
    violations_open: int = 0
    violations_by_severity: Dict[str, int] = field(default_factory=dict)
    risk_score: int = 0


class PoliciesAPI:
    """API interface for compliance policy management."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Policy CRUD
    # =========================================================================

    def list(
        self,
        workspace_id: Optional[str] = None,
        vertical_id: Optional[str] = None,
        framework_id: Optional[str] = None,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[Policy], int]:
        """
        List policies with optional filters.

        Args:
            workspace_id: Filter by workspace.
            vertical_id: Filter by vertical.
            framework_id: Filter by compliance framework.
            enabled_only: Only return enabled policies.
            limit: Maximum number of policies to return.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of Policy objects, total count).
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if vertical_id:
            params["vertical_id"] = vertical_id
        if framework_id:
            params["framework_id"] = framework_id
        if enabled_only:
            params["enabled_only"] = enabled_only

        response = self._client._get("/api/v1/policies", params=params)
        policies = [self._parse_policy(p) for p in response.get("policies", [])]
        return policies, response.get("total", len(policies))

    async def list_async(
        self,
        workspace_id: Optional[str] = None,
        vertical_id: Optional[str] = None,
        framework_id: Optional[str] = None,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[Policy], int]:
        """Async version of list()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if vertical_id:
            params["vertical_id"] = vertical_id
        if framework_id:
            params["framework_id"] = framework_id
        if enabled_only:
            params["enabled_only"] = enabled_only

        response = await self._client._get_async("/api/v1/policies", params=params)
        policies = [self._parse_policy(p) for p in response.get("policies", [])]
        return policies, response.get("total", len(policies))

    def get(self, policy_id: str) -> Policy:
        """
        Get policy details.

        Args:
            policy_id: The policy ID.

        Returns:
            Policy object.
        """
        response = self._client._get(f"/api/v1/policies/{policy_id}")
        return self._parse_policy(response.get("policy", response))

    async def get_async(self, policy_id: str) -> Policy:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/v1/policies/{policy_id}")
        return self._parse_policy(response.get("policy", response))

    def create(
        self,
        name: str,
        framework_id: str,
        vertical_id: str,
        description: str = "",
        workspace_id: str = "default",
        level: str = "recommended",
        enabled: bool = True,
        rules: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Policy:
        """
        Create a new policy.

        Args:
            name: Policy name.
            framework_id: Compliance framework ID.
            vertical_id: Vertical ID.
            description: Policy description.
            workspace_id: Workspace ID.
            level: Policy level (required, recommended, optional).
            enabled: Whether the policy is enabled.
            rules: List of policy rules.
            metadata: Additional metadata.

        Returns:
            Created Policy object.
        """
        body: Dict[str, Any] = {
            "name": name,
            "framework_id": framework_id,
            "vertical_id": vertical_id,
            "description": description,
            "workspace_id": workspace_id,
            "level": level,
            "enabled": enabled,
        }
        if rules:
            body["rules"] = rules
        if metadata:
            body["metadata"] = metadata

        response = self._client._post("/api/v1/policies", body)
        return self._parse_policy(response.get("policy", response))

    async def create_async(
        self,
        name: str,
        framework_id: str,
        vertical_id: str,
        description: str = "",
        workspace_id: str = "default",
        level: str = "recommended",
        enabled: bool = True,
        rules: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Policy:
        """Async version of create()."""
        body: Dict[str, Any] = {
            "name": name,
            "framework_id": framework_id,
            "vertical_id": vertical_id,
            "description": description,
            "workspace_id": workspace_id,
            "level": level,
            "enabled": enabled,
        }
        if rules:
            body["rules"] = rules
        if metadata:
            body["metadata"] = metadata

        response = await self._client._post_async("/api/v1/policies", body)
        return self._parse_policy(response.get("policy", response))

    def update(
        self,
        policy_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        level: Optional[str] = None,
        enabled: Optional[bool] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Policy:
        """
        Update a policy.

        Args:
            policy_id: The policy ID.
            name: New policy name.
            description: New description.
            level: New level.
            enabled: Enable/disable policy.
            rules: Updated rules.
            metadata: Updated metadata.

        Returns:
            Updated Policy object.
        """
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if level is not None:
            body["level"] = level
        if enabled is not None:
            body["enabled"] = enabled
        if rules is not None:
            body["rules"] = rules
        if metadata is not None:
            body["metadata"] = metadata

        response = self._client._patch(f"/api/v1/policies/{policy_id}", body)
        return self._parse_policy(response.get("policy", response))

    async def update_async(
        self,
        policy_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        level: Optional[str] = None,
        enabled: Optional[bool] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Policy:
        """Async version of update()."""
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if level is not None:
            body["level"] = level
        if enabled is not None:
            body["enabled"] = enabled
        if rules is not None:
            body["rules"] = rules
        if metadata is not None:
            body["metadata"] = metadata

        response = await self._client._patch_async(f"/api/v1/policies/{policy_id}", body)
        return self._parse_policy(response.get("policy", response))

    def delete(self, policy_id: str) -> bool:
        """
        Delete a policy.

        Args:
            policy_id: The policy ID.

        Returns:
            True if successful.
        """
        self._client._delete(f"/api/v1/policies/{policy_id}")
        return True

    async def delete_async(self, policy_id: str) -> bool:
        """Async version of delete()."""
        await self._client._delete_async(f"/api/v1/policies/{policy_id}")
        return True

    def toggle(self, policy_id: str, enabled: Optional[bool] = None) -> Policy:
        """
        Toggle a policy's enabled status.

        Args:
            policy_id: The policy ID.
            enabled: Explicit enabled state (if None, toggles current state).

        Returns:
            Updated Policy object.
        """
        body: Dict[str, Any] = {}
        if enabled is not None:
            body["enabled"] = enabled

        response = self._client._post(f"/api/v1/policies/{policy_id}/toggle", body)
        return self._parse_policy(response.get("policy", response))

    async def toggle_async(self, policy_id: str, enabled: Optional[bool] = None) -> Policy:
        """Async version of toggle()."""
        body: Dict[str, Any] = {}
        if enabled is not None:
            body["enabled"] = enabled

        response = await self._client._post_async(f"/api/v1/policies/{policy_id}/toggle", body)
        return self._parse_policy(response.get("policy", response))

    # =========================================================================
    # Violations
    # =========================================================================

    def list_violations(
        self,
        policy_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[PolicyViolation], int]:
        """
        List policy violations.

        Args:
            policy_id: Filter by policy.
            workspace_id: Filter by workspace.
            status: Filter by status.
            severity: Filter by severity.
            limit: Maximum number of violations to return.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of PolicyViolation objects, total count).
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if policy_id:
            params["policy_id"] = policy_id
        if workspace_id:
            params["workspace_id"] = workspace_id
        if status:
            params["status"] = status
        if severity:
            params["severity"] = severity

        response = self._client._get("/api/v1/compliance/violations", params=params)
        violations = [self._parse_violation(v) for v in response.get("violations", [])]
        return violations, response.get("total", len(violations))

    async def list_violations_async(
        self,
        policy_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[PolicyViolation], int]:
        """Async version of list_violations()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if policy_id:
            params["policy_id"] = policy_id
        if workspace_id:
            params["workspace_id"] = workspace_id
        if status:
            params["status"] = status
        if severity:
            params["severity"] = severity

        response = await self._client._get_async("/api/v1/compliance/violations", params=params)
        violations = [self._parse_violation(v) for v in response.get("violations", [])]
        return violations, response.get("total", len(violations))

    def get_violation(self, violation_id: str) -> PolicyViolation:
        """
        Get violation details.

        Args:
            violation_id: The violation ID.

        Returns:
            PolicyViolation object.
        """
        response = self._client._get(f"/api/v1/compliance/violations/{violation_id}")
        return self._parse_violation(response.get("violation", response))

    async def get_violation_async(self, violation_id: str) -> PolicyViolation:
        """Async version of get_violation()."""
        response = await self._client._get_async(f"/api/v1/compliance/violations/{violation_id}")
        return self._parse_violation(response.get("violation", response))

    def update_violation(
        self,
        violation_id: str,
        status: str,
        resolution_notes: Optional[str] = None,
    ) -> PolicyViolation:
        """
        Update violation status.

        Args:
            violation_id: The violation ID.
            status: New status (open, investigating, resolved, false_positive).
            resolution_notes: Notes about the resolution.

        Returns:
            Updated PolicyViolation object.
        """
        body: Dict[str, Any] = {"status": status}
        if resolution_notes:
            body["resolution_notes"] = resolution_notes

        response = self._client._patch(f"/api/v1/compliance/violations/{violation_id}", body)
        return self._parse_violation(response.get("violation", response))

    async def update_violation_async(
        self,
        violation_id: str,
        status: str,
        resolution_notes: Optional[str] = None,
    ) -> PolicyViolation:
        """Async version of update_violation()."""
        body: Dict[str, Any] = {"status": status}
        if resolution_notes:
            body["resolution_notes"] = resolution_notes

        response = await self._client._patch_async(
            f"/api/v1/compliance/violations/{violation_id}", body
        )
        return self._parse_violation(response.get("violation", response))

    # =========================================================================
    # Compliance Checking
    # =========================================================================

    def check(
        self,
        content: str,
        frameworks: Optional[List[str]] = None,
        min_severity: str = "low",
        store_violations: bool = False,
        workspace_id: str = "default",
    ) -> ComplianceCheckResult:
        """
        Run compliance check on content.

        Args:
            content: Content to check.
            frameworks: Specific frameworks to check against.
            min_severity: Minimum severity to report.
            store_violations: Whether to store detected violations.
            workspace_id: Workspace ID for storing violations.

        Returns:
            ComplianceCheckResult object.
        """
        body: Dict[str, Any] = {
            "content": content,
            "min_severity": min_severity,
            "store_violations": store_violations,
            "workspace_id": workspace_id,
        }
        if frameworks:
            body["frameworks"] = frameworks

        response = self._client._post("/api/v1/compliance/check", body)
        return self._parse_check_result(response)

    async def check_async(
        self,
        content: str,
        frameworks: Optional[List[str]] = None,
        min_severity: str = "low",
        store_violations: bool = False,
        workspace_id: str = "default",
    ) -> ComplianceCheckResult:
        """Async version of check()."""
        body: Dict[str, Any] = {
            "content": content,
            "min_severity": min_severity,
            "store_violations": store_violations,
            "workspace_id": workspace_id,
        }
        if frameworks:
            body["frameworks"] = frameworks

        response = await self._client._post_async("/api/v1/compliance/check", body)
        return self._parse_check_result(response)

    def get_stats(self, workspace_id: Optional[str] = None) -> ComplianceStats:
        """
        Get compliance statistics.

        Args:
            workspace_id: Filter by workspace.

        Returns:
            ComplianceStats object.
        """
        params: Dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id

        response = self._client._get("/api/v1/compliance/stats", params=params)
        return self._parse_stats(response)

    async def get_stats_async(self, workspace_id: Optional[str] = None) -> ComplianceStats:
        """Async version of get_stats()."""
        params: Dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id

        response = await self._client._get_async("/api/v1/compliance/stats", params=params)
        return self._parse_stats(response)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_policy(self, data: Dict[str, Any]) -> Policy:
        """Parse policy data into Policy object."""
        created_at = None
        updated_at = None

        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("updated_at"):
            try:
                updated_at = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        rules = [self._parse_rule(r) for r in data.get("rules", [])]

        return Policy(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            framework_id=data.get("framework_id", ""),
            workspace_id=data.get("workspace_id", "default"),
            vertical_id=data.get("vertical_id", ""),
            level=data.get("level", "recommended"),
            enabled=data.get("enabled", True),
            rules=rules,
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get("metadata", {}),
        )

    def _parse_rule(self, data: Dict[str, Any]) -> PolicyRule:
        """Parse rule data into PolicyRule object."""
        return PolicyRule(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            condition=data.get("condition", ""),
            severity=data.get("severity", "medium"),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )

    def _parse_violation(self, data: Dict[str, Any]) -> PolicyViolation:
        """Parse violation data into PolicyViolation object."""
        created_at = None
        resolved_at = None

        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("resolved_at"):
            try:
                resolved_at = datetime.fromisoformat(data["resolved_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return PolicyViolation(
            id=data.get("id", ""),
            policy_id=data.get("policy_id", ""),
            rule_id=data.get("rule_id", ""),
            rule_name=data.get("rule_name", ""),
            framework_id=data.get("framework_id", ""),
            vertical_id=data.get("vertical_id", ""),
            workspace_id=data.get("workspace_id", "default"),
            severity=data.get("severity", "medium"),
            status=data.get("status", "open"),
            description=data.get("description", ""),
            source=data.get("source", ""),
            created_at=created_at,
            resolved_at=resolved_at,
            resolved_by=data.get("resolved_by"),
            resolution_notes=data.get("resolution_notes"),
            metadata=data.get("metadata", {}),
        )

    def _parse_check_result(self, data: Dict[str, Any]) -> ComplianceCheckResult:
        """Parse check result data into ComplianceCheckResult object."""
        checked_at = None
        if data.get("checked_at"):
            try:
                checked_at = datetime.fromisoformat(data["checked_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        result_data = data.get("result", data)
        return ComplianceCheckResult(
            compliant=result_data.get("compliant", data.get("compliant", True)),
            score=result_data.get("score", data.get("score", 100.0)),
            issues=result_data.get("issues", data.get("issues", [])),
            checked_at=checked_at,
        )

    def _parse_stats(self, data: Dict[str, Any]) -> ComplianceStats:
        """Parse stats data into ComplianceStats object."""
        policies = data.get("policies", {})
        violations = data.get("violations", {})

        return ComplianceStats(
            policies_total=policies.get("total", 0),
            policies_enabled=policies.get("enabled", 0),
            policies_disabled=policies.get("disabled", 0),
            violations_total=violations.get("total", 0),
            violations_open=violations.get("open", 0),
            violations_by_severity=violations.get("by_severity", {}),
            risk_score=data.get("risk_score", 0),
        )


__all__ = [
    "PoliciesAPI",
    "Policy",
    "PolicyRule",
    "PolicyViolation",
    "ComplianceCheckResult",
    "ComplianceStats",
]
