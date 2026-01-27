"""
Unified Consistency Validator for Knowledge Mound.

Aggregates all consistency checking capabilities into a single interface:
- Referential integrity (orphaned nodes, broken links)
- Content validation (size, format, schema)
- Contradiction detection (semantic, logical, temporal)
- Staleness detection (outdated content)
- Confidence decay validation
- Adapter synchronization status

Provides a single API for comprehensive consistency assessment.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from aragora.knowledge.mound.facade import KnowledgeMound

logger = logging.getLogger(__name__)


class ConsistencyCheckType(str, Enum):
    """Types of consistency checks available."""

    REFERENTIAL = "referential"  # Orphaned nodes, broken links
    CONTENT = "content"  # Size, format, schema validation
    CONTRADICTION = "contradiction"  # Conflicting knowledge items
    STALENESS = "staleness"  # Outdated content detection
    CONFIDENCE = "confidence"  # Confidence decay validation
    SYNC = "sync"  # Adapter synchronization status
    ALL = "all"  # Run all checks


class ConsistencySeverity(str, Enum):
    """Severity levels for consistency issues."""

    CRITICAL = "critical"  # Data corruption, immediate action required
    HIGH = "high"  # Significant issues affecting reliability
    MEDIUM = "medium"  # Issues that should be addressed
    LOW = "low"  # Minor issues, cosmetic
    INFO = "info"  # Informational, no action required


@dataclass
class ConsistencyIssue:
    """A single consistency issue found during validation."""

    check_type: ConsistencyCheckType
    severity: ConsistencySeverity
    message: str
    item_id: Optional[str] = None
    related_items: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "check_type": self.check_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "item_id": self.item_id,
            "related_items": self.related_items,
            "details": self.details,
            "suggested_fix": self.suggested_fix,
            "auto_fixable": self.auto_fixable,
        }


@dataclass
class ConsistencyCheckResult:
    """Result of a single consistency check."""

    check_type: ConsistencyCheckType
    passed: bool
    items_checked: int
    issues_found: int
    issues: List[ConsistencyIssue] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "check_type": self.check_type.value,
            "passed": self.passed,
            "items_checked": self.items_checked,
            "issues_found": self.issues_found,
            "issues": [i.to_dict() for i in self.issues],
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


@dataclass
class ConsistencyReport:
    """Complete consistency validation report."""

    workspace_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    overall_healthy: bool = True
    checks_run: List[ConsistencyCheckResult] = field(default_factory=list)
    total_items_checked: int = 0
    total_issues_found: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workspace_id": self.workspace_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_healthy": self.overall_healthy,
            "summary": {
                "total_items_checked": self.total_items_checked,
                "total_issues_found": self.total_issues_found,
                "critical": self.critical_issues,
                "high": self.high_issues,
                "medium": self.medium_issues,
                "low": self.low_issues,
            },
            "checks": [c.to_dict() for c in self.checks_run],
            "duration_ms": round(self.duration_ms, 2),
        }

    def add_result(self, result: ConsistencyCheckResult) -> None:
        """Add a check result and update summary."""
        self.checks_run.append(result)
        self.total_items_checked += result.items_checked
        self.total_issues_found += result.issues_found
        self.duration_ms += result.duration_ms

        for issue in result.issues:
            if issue.severity == ConsistencySeverity.CRITICAL:
                self.critical_issues += 1
            elif issue.severity == ConsistencySeverity.HIGH:
                self.high_issues += 1
            elif issue.severity == ConsistencySeverity.MEDIUM:
                self.medium_issues += 1
            elif issue.severity == ConsistencySeverity.LOW:
                self.low_issues += 1

        # Overall health is false if any critical/high issues or check failures
        if not result.passed or result.error:
            self.overall_healthy = False
        if self.critical_issues > 0 or self.high_issues > 0:
            self.overall_healthy = False


class ConsistencyValidator:
    """
    Unified consistency validator for Knowledge Mound.

    Orchestrates multiple consistency checks and produces a comprehensive report.

    Usage:
        validator = ConsistencyValidator(knowledge_mound)
        report = await validator.validate("workspace_123")
        if not report.overall_healthy:
            print(f"Found {report.total_issues_found} issues")
    """

    def __init__(
        self,
        mound: "KnowledgeMound",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize consistency validator.

        Args:
            mound: Knowledge Mound instance to validate
            config: Optional configuration overrides
        """
        self._mound = mound
        self._config = config or {}

        # Default configuration
        self._max_stale_days = self._config.get("max_stale_days", 90)
        self._min_confidence = self._config.get("min_confidence", 0.3)
        self._contradiction_threshold = self._config.get("contradiction_threshold", 0.7)
        self._max_content_size = self._config.get("max_content_size", 100_000)

    async def validate(
        self,
        workspace_id: str,
        check_types: Optional[List[ConsistencyCheckType]] = None,
    ) -> ConsistencyReport:
        """
        Run consistency validation for a workspace.

        Args:
            workspace_id: Workspace to validate
            check_types: Specific checks to run (default: all)

        Returns:
            ConsistencyReport with all findings
        """
        import time

        start_time = time.perf_counter()
        report = ConsistencyReport(workspace_id=workspace_id)

        if check_types is None or ConsistencyCheckType.ALL in check_types:
            check_types = [
                ConsistencyCheckType.REFERENTIAL,
                ConsistencyCheckType.CONTENT,
                ConsistencyCheckType.CONTRADICTION,
                ConsistencyCheckType.STALENESS,
                ConsistencyCheckType.CONFIDENCE,
                ConsistencyCheckType.SYNC,
            ]

        # Run checks in parallel where possible
        check_tasks = []
        for check_type in check_types:
            if check_type == ConsistencyCheckType.REFERENTIAL:
                check_tasks.append(self._check_referential_integrity(workspace_id))
            elif check_type == ConsistencyCheckType.CONTENT:
                check_tasks.append(self._check_content_validation(workspace_id))
            elif check_type == ConsistencyCheckType.CONTRADICTION:
                check_tasks.append(self._check_contradictions(workspace_id))
            elif check_type == ConsistencyCheckType.STALENESS:
                check_tasks.append(self._check_staleness(workspace_id))
            elif check_type == ConsistencyCheckType.CONFIDENCE:
                check_tasks.append(self._check_confidence_decay(workspace_id))
            elif check_type == ConsistencyCheckType.SYNC:
                check_tasks.append(self._check_adapter_sync(workspace_id))

        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Consistency check failed: {result}")
                error_result = ConsistencyCheckResult(
                    check_type=ConsistencyCheckType.ALL,
                    passed=False,
                    items_checked=0,
                    issues_found=0,
                    error=str(result),
                )
                report.add_result(error_result)
            else:
                report.add_result(result)

        report.duration_ms = (time.perf_counter() - start_time) * 1000
        return report

    async def _check_referential_integrity(self, workspace_id: str) -> ConsistencyCheckResult:
        """Check for orphaned nodes and broken links."""
        import time

        start_time = time.perf_counter()
        issues: List[ConsistencyIssue] = []
        items_checked = 0

        try:
            # Get all nodes
            nodes = await self._mound.query(workspace_id, "", limit=10000)
            items_checked = len(nodes)

            node_ids: Set[str] = {n.get("id", n.get("node_id", "")) for n in nodes if n}
            referenced_ids: Set[str] = set()

            # Check for broken references
            for node in nodes:
                if not node:
                    continue

                node_id = node.get("id", node.get("node_id", ""))

                # Check parent references
                parent_id = node.get("parent_id")
                if parent_id and parent_id not in node_ids:
                    issues.append(
                        ConsistencyIssue(
                            check_type=ConsistencyCheckType.REFERENTIAL,
                            severity=ConsistencySeverity.HIGH,
                            message=f"Broken parent reference: {parent_id}",
                            item_id=node_id,
                            related_items=[parent_id],
                            suggested_fix="Remove parent reference or restore parent node",
                            auto_fixable=True,
                        )
                    )

                # Check relationship references
                relationships = node.get("relationships", [])
                for rel in relationships:
                    target_id = rel.get("target_id")
                    if target_id and target_id not in node_ids:
                        issues.append(
                            ConsistencyIssue(
                                check_type=ConsistencyCheckType.REFERENTIAL,
                                severity=ConsistencySeverity.MEDIUM,
                                message=f"Broken relationship to: {target_id}",
                                item_id=node_id,
                                related_items=[target_id],
                                suggested_fix="Remove broken relationship",
                                auto_fixable=True,
                            )
                        )
                    else:
                        referenced_ids.add(target_id)

            # Check for orphaned nodes (no references to them)
            orphaned = node_ids - referenced_ids
            # Filter out root nodes (nodes without parent_id are allowed)
            for node in nodes:
                if not node:
                    continue
                node_id = node.get("id", node.get("node_id", ""))
                if node_id in orphaned and not node.get("parent_id"):
                    orphaned.discard(node_id)

            for orphan_id in orphaned:
                issues.append(
                    ConsistencyIssue(
                        check_type=ConsistencyCheckType.REFERENTIAL,
                        severity=ConsistencySeverity.LOW,
                        message=f"Potentially orphaned node: {orphan_id}",
                        item_id=orphan_id,
                        suggested_fix="Review and link to knowledge graph or delete",
                    )
                )

        except Exception as e:
            logger.error(f"Referential integrity check failed: {e}")
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.REFERENTIAL,
                passed=False,
                items_checked=items_checked,
                issues_found=len(issues),
                issues=issues,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.REFERENTIAL,
            passed=len(
                [
                    i
                    for i in issues
                    if i.severity in [ConsistencySeverity.CRITICAL, ConsistencySeverity.HIGH]
                ]
            )
            == 0,
            items_checked=items_checked,
            issues_found=len(issues),
            issues=issues,
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def _check_content_validation(self, workspace_id: str) -> ConsistencyCheckResult:
        """Check content size, format, and schema validation."""
        import time

        start_time = time.perf_counter()
        issues: List[ConsistencyIssue] = []
        items_checked = 0

        try:
            nodes = await self._mound.query(workspace_id, "", limit=10000)
            items_checked = len(nodes)

            for node in nodes:
                if not node:
                    continue

                node_id = node.get("id", node.get("node_id", ""))
                content = node.get("content", "")

                # Check content size
                if len(content) > self._max_content_size:
                    issues.append(
                        ConsistencyIssue(
                            check_type=ConsistencyCheckType.CONTENT,
                            severity=ConsistencySeverity.MEDIUM,
                            message=f"Content exceeds max size ({len(content)} > {self._max_content_size})",
                            item_id=node_id,
                            details={
                                "content_size": len(content),
                                "max_size": self._max_content_size,
                            },
                            suggested_fix="Split into smaller nodes or summarize",
                        )
                    )

                # Check for empty content
                if not content or not content.strip():
                    issues.append(
                        ConsistencyIssue(
                            check_type=ConsistencyCheckType.CONTENT,
                            severity=ConsistencySeverity.LOW,
                            message="Empty content",
                            item_id=node_id,
                            suggested_fix="Add content or delete empty node",
                            auto_fixable=True,
                        )
                    )

                # Check required fields
                required_fields = ["id", "workspace_id"]
                for req_field in required_fields:
                    if req_field not in node or not node.get(req_field):
                        issues.append(
                            ConsistencyIssue(
                                check_type=ConsistencyCheckType.CONTENT,
                                severity=ConsistencySeverity.HIGH,
                                message=f"Missing required field: {req_field}",
                                item_id=node_id,
                                suggested_fix=f"Add {req_field} to node",
                            )
                        )

        except Exception as e:
            logger.error(f"Content validation check failed: {e}")
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.CONTENT,
                passed=False,
                items_checked=items_checked,
                issues_found=len(issues),
                issues=issues,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.CONTENT,
            passed=len(
                [
                    i
                    for i in issues
                    if i.severity in [ConsistencySeverity.CRITICAL, ConsistencySeverity.HIGH]
                ]
            )
            == 0,
            items_checked=items_checked,
            issues_found=len(issues),
            issues=issues,
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def _check_contradictions(self, workspace_id: str) -> ConsistencyCheckResult:
        """Check for contradicting knowledge items using existing ContradictionDetector."""
        import time

        start_time = time.perf_counter()
        issues: List[ConsistencyIssue] = []
        items_checked = 0

        try:
            # Use existing contradiction detector if available
            from aragora.knowledge.mound.ops.contradiction import ContradictionDetector

            detector = ContradictionDetector()
            report = await detector.detect_contradictions(
                self._mound,
                workspace_id,
            )

            items_checked = report.scanned_items

            for contradiction in report.contradictions:
                severity = ConsistencySeverity.MEDIUM
                if contradiction.severity == "critical":
                    severity = ConsistencySeverity.CRITICAL
                elif contradiction.severity == "high":
                    severity = ConsistencySeverity.HIGH
                elif contradiction.severity == "low":
                    severity = ConsistencySeverity.LOW

                issues.append(
                    ConsistencyIssue(
                        check_type=ConsistencyCheckType.CONTRADICTION,
                        severity=severity,
                        message=f"Contradiction detected ({contradiction.contradiction_type.value})",
                        item_id=contradiction.item_a_id,
                        related_items=[contradiction.item_b_id],
                        details={
                            "similarity_score": contradiction.similarity_score,
                            "conflict_score": contradiction.conflict_score,
                            "type": contradiction.contradiction_type.value,
                        },
                        suggested_fix=f"Review and resolve using: {contradiction.resolution.value if contradiction.resolution else 'human_review'}",
                    )
                )

        except ImportError:
            logger.warning("ContradictionDetector not available, skipping contradiction check")
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.CONTRADICTION,
                passed=True,
                items_checked=0,
                issues_found=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Contradiction check failed: {e}")
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.CONTRADICTION,
                passed=False,
                items_checked=items_checked,
                issues_found=len(issues),
                issues=issues,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.CONTRADICTION,
            passed=len(
                [
                    i
                    for i in issues
                    if i.severity in [ConsistencySeverity.CRITICAL, ConsistencySeverity.HIGH]
                ]
            )
            == 0,
            items_checked=items_checked,
            issues_found=len(issues),
            issues=issues,
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def _check_staleness(self, workspace_id: str) -> ConsistencyCheckResult:
        """Check for stale/outdated content."""
        import time

        start_time = time.perf_counter()
        issues: List[ConsistencyIssue] = []
        items_checked = 0

        try:
            nodes = await self._mound.query(workspace_id, "", limit=10000)
            items_checked = len(nodes)
            cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_stale_days)

            for node in nodes:
                if not node:
                    continue

                node_id = node.get("id", node.get("node_id", ""))
                updated_at = node.get("updated_at")

                if updated_at:
                    # Parse timestamp
                    if isinstance(updated_at, str):
                        try:
                            updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        except ValueError:
                            continue
                    elif isinstance(updated_at, datetime):
                        updated_dt = updated_at
                    else:
                        continue

                    # Make timezone aware if needed
                    if updated_dt.tzinfo is None:
                        updated_dt = updated_dt.replace(tzinfo=timezone.utc)

                    if updated_dt < cutoff:
                        days_stale = (datetime.now(timezone.utc) - updated_dt).days
                        issues.append(
                            ConsistencyIssue(
                                check_type=ConsistencyCheckType.STALENESS,
                                severity=ConsistencySeverity.LOW
                                if days_stale < 180
                                else ConsistencySeverity.MEDIUM,
                                message=f"Content not updated in {days_stale} days",
                                item_id=node_id,
                                details={"days_stale": days_stale, "last_updated": updated_at},
                                suggested_fix="Review and update content or mark as archived",
                            )
                        )

        except Exception as e:
            logger.error(f"Staleness check failed: {e}")
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.STALENESS,
                passed=False,
                items_checked=items_checked,
                issues_found=len(issues),
                issues=issues,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.STALENESS,
            passed=True,  # Staleness doesn't fail the check, just reports
            items_checked=items_checked,
            issues_found=len(issues),
            issues=issues,
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def _check_confidence_decay(self, workspace_id: str) -> ConsistencyCheckResult:
        """Check for low confidence items that need review."""
        import time

        start_time = time.perf_counter()
        issues: List[ConsistencyIssue] = []
        items_checked = 0

        try:
            nodes = await self._mound.query(workspace_id, "", limit=10000)
            items_checked = len(nodes)

            for node in nodes:
                if not node:
                    continue

                node_id = node.get("id", node.get("node_id", ""))
                confidence = node.get("confidence", node.get("confidence_score", 1.0))

                if confidence is not None and confidence < self._min_confidence:
                    issues.append(
                        ConsistencyIssue(
                            check_type=ConsistencyCheckType.CONFIDENCE,
                            severity=ConsistencySeverity.LOW
                            if confidence > 0.1
                            else ConsistencySeverity.MEDIUM,
                            message=f"Low confidence score: {confidence:.2f}",
                            item_id=node_id,
                            details={
                                "confidence": confidence,
                                "min_confidence": self._min_confidence,
                            },
                            suggested_fix="Re-validate content or add supporting evidence",
                        )
                    )

        except Exception as e:
            logger.error(f"Confidence decay check failed: {e}")
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.CONFIDENCE,
                passed=False,
                items_checked=items_checked,
                issues_found=len(issues),
                issues=issues,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.CONFIDENCE,
            passed=True,  # Low confidence doesn't fail, just reports
            items_checked=items_checked,
            issues_found=len(issues),
            issues=issues,
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def _check_adapter_sync(self, workspace_id: str) -> ConsistencyCheckResult:
        """Check adapter synchronization status."""
        import time

        start_time = time.perf_counter()
        issues: List[ConsistencyIssue] = []

        try:
            # Get adapter status
            adapters = self._mound.list_adapters() if hasattr(self._mound, "list_adapters") else []
            items_checked = len(adapters)

            for adapter_name in adapters:
                try:
                    adapter = (
                        self._mound.get_adapter(adapter_name)
                        if hasattr(self._mound, "get_adapter")
                        else None
                    )
                    if adapter and hasattr(adapter, "get_health"):
                        health = await adapter.get_health()
                        if not health.get("healthy", True):
                            issues.append(
                                ConsistencyIssue(
                                    check_type=ConsistencyCheckType.SYNC,
                                    severity=ConsistencySeverity.HIGH,
                                    message=f"Adapter unhealthy: {adapter_name}",
                                    details=health,
                                    suggested_fix="Check adapter connection and restart if needed",
                                )
                            )
                except Exception as adapter_error:
                    issues.append(
                        ConsistencyIssue(
                            check_type=ConsistencyCheckType.SYNC,
                            severity=ConsistencySeverity.MEDIUM,
                            message=f"Could not check adapter: {adapter_name}",
                            details={"error": str(adapter_error)},
                        )
                    )

        except Exception as e:
            logger.error(f"Adapter sync check failed: {e}")
            return ConsistencyCheckResult(
                check_type=ConsistencyCheckType.SYNC,
                passed=False,
                items_checked=0,
                issues_found=len(issues),
                issues=issues,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

        return ConsistencyCheckResult(
            check_type=ConsistencyCheckType.SYNC,
            passed=len(
                [
                    i
                    for i in issues
                    if i.severity in [ConsistencySeverity.CRITICAL, ConsistencySeverity.HIGH]
                ]
            )
            == 0,
            items_checked=items_checked,
            issues_found=len(issues),
            issues=issues,
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def auto_fix(
        self,
        workspace_id: str,
        issue_types: Optional[List[ConsistencyCheckType]] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Attempt to auto-fix consistency issues.

        Args:
            workspace_id: Workspace to fix
            issue_types: Types of issues to fix (default: all auto-fixable)
            dry_run: If True, only report what would be fixed

        Returns:
            Summary of fixes applied (or would be applied in dry run)
        """
        report = await self.validate(workspace_id, issue_types)
        fixes_applied = []
        fixes_failed = []

        for check in report.checks_run:
            for issue in check.issues:
                if not issue.auto_fixable:
                    continue

                if dry_run:
                    fixes_applied.append(
                        {
                            "issue": issue.to_dict(),
                            "action": "would_fix",
                        }
                    )
                else:
                    try:
                        # Apply fix based on issue type
                        if issue.check_type == ConsistencyCheckType.REFERENTIAL:
                            if "Broken parent reference" in issue.message:
                                await self._mound.update(
                                    workspace_id,
                                    issue.item_id,
                                    {"parent_id": None},
                                )
                            elif "Broken relationship" in issue.message:
                                # Remove broken relationship
                                node = await self._mound.get(workspace_id, issue.item_id)
                                if node:
                                    relationships = node.get("relationships", [])
                                    relationships = [
                                        r
                                        for r in relationships
                                        if r.get("target_id") not in issue.related_items
                                    ]
                                    await self._mound.update(
                                        workspace_id,
                                        issue.item_id,
                                        {"relationships": relationships},
                                    )

                        elif issue.check_type == ConsistencyCheckType.CONTENT:
                            if "Empty content" in issue.message:
                                await self._mound.delete(workspace_id, issue.item_id)

                        fixes_applied.append(
                            {
                                "issue": issue.to_dict(),
                                "action": "fixed",
                            }
                        )

                    except Exception as e:
                        fixes_failed.append(
                            {
                                "issue": issue.to_dict(),
                                "error": str(e),
                            }
                        )

        return {
            "workspace_id": workspace_id,
            "dry_run": dry_run,
            "total_auto_fixable": len(
                [i for c in report.checks_run for i in c.issues if i.auto_fixable]
            ),
            "fixes_applied": len(fixes_applied),
            "fixes_failed": len(fixes_failed),
            "details": {
                "applied": fixes_applied,
                "failed": fixes_failed,
            },
        }


__all__ = [
    "ConsistencyValidator",
    "ConsistencyCheckType",
    "ConsistencySeverity",
    "ConsistencyIssue",
    "ConsistencyCheckResult",
    "ConsistencyReport",
]
