"""Gauntlet → Self-Improvement Bridge.

Converts Gauntlet findings into actionable self-improvement goals
that the Nomic Loop can execute. Closes the loop:

    Gauntlet finds weakness → Bridge creates goal → Nomic Loop fixes it

Usage:
    from aragora.gauntlet.improvement_bridge import findings_to_goals

    goals = findings_to_goals(gauntlet_result)
    for goal in goals:
        orchestrator.execute_goal(goal.description)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Severity → priority mapping (lower = higher priority)
_SEVERITY_PRIORITY = {
    "critical": 1,
    "high": 2,
    "medium": 3,
    "low": 4,
    "info": 5,
}

# Category → track mapping for Nomic Loop routing
_CATEGORY_TRACK = {
    "security": "security",
    "authentication": "security",
    "authorization": "security",
    "injection": "security",
    "xss": "security",
    "encryption": "security",
    "performance": "core",
    "reliability": "core",
    "error_handling": "core",
    "validation": "core",
    "documentation": "developer",
    "api_design": "developer",
    "test_coverage": "qa",
    "regression": "qa",
    "usability": "sme",
    "accessibility": "sme",
}


@dataclass
class ImprovementGoal:
    """A self-improvement goal derived from Gauntlet findings."""

    id: str
    description: str
    priority: int  # 1 = highest
    track: str  # Nomic Loop track (security, core, developer, qa, sme)
    source_finding_id: str
    severity: str
    category: str
    file_hints: list[str] = field(default_factory=list)
    success_criteria: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "track": self.track,
            "source_finding_id": self.source_finding_id,
            "severity": self.severity,
            "category": self.category,
            "file_hints": self.file_hints,
            "success_criteria": self.success_criteria,
        }


def findings_to_goals(
    gauntlet_result: Any,
    max_goals: int = 10,
    min_severity: str = "medium",
) -> list[ImprovementGoal]:
    """Convert Gauntlet findings into prioritized self-improvement goals.

    Args:
        gauntlet_result: A GauntletResult (from config.py, result.py, or orchestrator.py)
        max_goals: Maximum goals to generate
        min_severity: Minimum severity to include

    Returns:
        List of ImprovementGoal sorted by priority (highest first)
    """
    findings = _extract_findings(gauntlet_result)
    if not findings:
        return []

    # Filter by minimum severity
    min_priority = _SEVERITY_PRIORITY.get(min_severity, 3)
    filtered = [
        f for f in findings
        if _SEVERITY_PRIORITY.get(_get_severity(f), 5) <= min_priority
    ]

    goals: list[ImprovementGoal] = []
    for i, finding in enumerate(filtered[:max_goals]):
        finding_id = _get_field(finding, "id", f"finding-{i}")
        severity = _get_severity(finding)
        category = _get_field(finding, "category", "general")
        description = _get_field(finding, "description", "") or _get_field(finding, "title", "")
        recommendation = _get_field(finding, "recommendation", "")

        if not description:
            continue

        # Build improvement description
        goal_desc = f"Fix {severity} {category} issue: {description}"
        if recommendation:
            goal_desc += f". Recommended fix: {recommendation}"

        # Extract file hints
        file_hints: list[str] = []
        location = _get_field(finding, "location", "")
        if location:
            file_hints.append(location)
        affected_files = _get_field(finding, "affected_files", [])
        if isinstance(affected_files, list):
            file_hints.extend(affected_files)

        goal = ImprovementGoal(
            id=f"gauntlet-{finding_id}",
            description=goal_desc,
            priority=_SEVERITY_PRIORITY.get(severity, 3),
            track=_CATEGORY_TRACK.get(category, "core"),
            source_finding_id=finding_id,
            severity=severity,
            category=category,
            file_hints=file_hints,
            success_criteria={"gauntlet_retest": "pass"},
        )
        goals.append(goal)

    # Sort by priority
    goals.sort(key=lambda g: g.priority)
    logger.info(
        "gauntlet_to_goals findings=%d goals=%d top_severity=%s",
        len(findings), len(goals),
        goals[0].severity if goals else "none",
    )
    return goals


def _extract_findings(result: Any) -> list[Any]:
    """Extract findings from various GauntletResult formats."""
    # Try direct .findings attribute
    findings = getattr(result, "findings", None)
    if findings and isinstance(findings, list):
        return findings

    # Try .all_findings() method
    all_findings = getattr(result, "all_findings", None)
    if callable(all_findings):
        try:
            return all_findings()
        except (TypeError, RuntimeError):
            pass

    # Try dict access
    if isinstance(result, dict):
        return result.get("findings", [])

    return []


def _get_severity(finding: Any) -> str:
    """Extract severity string from a finding."""
    severity = getattr(finding, "severity", None)
    if severity is None and isinstance(finding, dict):
        severity = finding.get("severity", "medium")
    if hasattr(severity, "value"):  # Enum
        return str(severity.value).lower()
    return str(severity).lower() if severity else "medium"


def _get_field(finding: Any, field_name: str, default: Any = "") -> Any:
    """Get a field from finding (supports both dataclass and dict)."""
    val = getattr(finding, field_name, None)
    if val is not None:
        return val
    if isinstance(finding, dict):
        return finding.get(field_name, default)
    return default
