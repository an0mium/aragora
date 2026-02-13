"""
ComplianceAdapter - Bridges compliance check results and violations to the Knowledge Mound.

This adapter persists compliance check results, violation lifecycle data,
and framework health snapshots for trend analysis and institutional learning.

The adapter provides:
- Compliance check result persistence with issue details
- Violation lifecycle tracking (detected -> resolved)
- Framework health scoring over time
- Trend analysis for compliance posture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem
    from aragora.compliance.framework import ComplianceCheckResult
    from aragora.compliance.policy_store import Violation

EventCallback = Callable[[str, dict[str, Any]], None]

logger = logging.getLogger(__name__)

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.adapters._semantic_mixin import SemanticSearchMixin
from aragora.knowledge.mound.adapters._reverse_flow_base import ReverseFlowMixin
from aragora.knowledge.mound.adapters._fusion_mixin import FusionMixin
from aragora.knowledge.mound.adapters._types import SyncResult


@dataclass
class ComplianceSearchResult:
    """Wrapper for compliance search results."""

    record_id: str
    record_type: str  # "check" or "violation"
    framework: str
    score: float
    severity: str = ""
    status: str = ""
    similarity: float = 0.0
    description: str = ""


@dataclass
class CheckOutcome:
    """Lightweight representation of a compliance check for adapter storage."""

    check_id: str
    compliant: bool
    score: float
    frameworks_checked: list[str] = field(default_factory=list)
    issue_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    issues_summary: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_check_result(
        cls, result: ComplianceCheckResult, check_id: str = ""
    ) -> CheckOutcome:
        """Create a CheckOutcome from a ComplianceCheckResult."""
        issues_summary = []
        critical_count = 0
        high_count = 0
        for issue in getattr(result, "issues", []):
            severity = str(getattr(issue, "severity", ""))
            if "critical" in severity.lower():
                critical_count += 1
            elif "high" in severity.lower():
                high_count += 1
            issues_summary.append(
                {
                    "framework": getattr(issue, "framework", ""),
                    "rule_id": getattr(issue, "rule_id", ""),
                    "severity": severity,
                    "description": getattr(issue, "description", "")[:200],
                }
            )

        return cls(
            check_id=check_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            compliant=getattr(result, "compliant", True),
            score=getattr(result, "score", 1.0),
            frameworks_checked=getattr(result, "frameworks_checked", []),
            issue_count=len(issues_summary),
            critical_count=critical_count,
            high_count=high_count,
            issues_summary=issues_summary,
        )


@dataclass
class ViolationOutcome:
    """Lightweight representation of a compliance violation for adapter storage."""

    violation_id: str
    policy_id: str
    rule_id: str
    rule_name: str
    framework_id: str
    severity: str
    status: str
    description: str
    source: str = ""
    workspace_id: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    resolution_notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_violation(cls, violation: Violation) -> ViolationOutcome:
        """Create a ViolationOutcome from a Violation."""
        return cls(
            violation_id=getattr(violation, "id", ""),
            policy_id=getattr(violation, "policy_id", ""),
            rule_id=getattr(violation, "rule_id", ""),
            rule_name=getattr(violation, "rule_name", ""),
            framework_id=getattr(violation, "framework_id", ""),
            severity=getattr(violation, "severity", "medium"),
            status=getattr(violation, "status", "open"),
            description=getattr(violation, "description", ""),
            source=getattr(violation, "source", ""),
            workspace_id=getattr(violation, "workspace_id", ""),
            detected_at=getattr(violation, "detected_at", datetime.now(timezone.utc)),
            resolved_at=getattr(violation, "resolved_at", None),
            resolution_notes=getattr(violation, "resolution_notes", None),
        )


class ComplianceAdapter(FusionMixin, ReverseFlowMixin, SemanticSearchMixin, KnowledgeMoundAdapter):
    """
    Adapter that bridges compliance results and violations to the Knowledge Mound.

    Provides compliance check persistence, violation lifecycle tracking, and
    framework health analytics for trend analysis.

    Usage:
        adapter = ComplianceAdapter()
        adapter.store_check(check_result, check_id="scan_001")
        adapter.store_violation(violation)
        await adapter.sync_to_km(mound)
        results = await adapter.search_violations(framework="soc2")
    """

    adapter_name = "compliance"
    source_type = "compliance"

    def __init__(
        self,
        enable_dual_write: bool = False,
        event_callback: EventCallback | None = None,
        enable_resilience: bool = True,
    ):
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_resilience=enable_resilience,
        )
        self._pending_checks: list[CheckOutcome] = []
        self._pending_violations: list[ViolationOutcome] = []
        self._synced_checks: dict[str, CheckOutcome] = {}
        self._synced_violations: dict[str, ViolationOutcome] = {}

    def store_check(self, result: Any, check_id: str = "") -> None:
        """Store a compliance check result for KM sync.

        Args:
            result: A ComplianceCheckResult or CheckOutcome.
            check_id: Optional ID for the check.
        """
        if isinstance(result, CheckOutcome):
            outcome = result
        else:
            outcome = CheckOutcome.from_check_result(result, check_id=check_id)

        outcome.metadata["km_sync_pending"] = True
        outcome.metadata["km_sync_requested_at"] = datetime.now(timezone.utc).isoformat()
        self._pending_checks.append(outcome)

        self._emit_event(
            "km_adapter_forward_sync",
            {
                "adapter": self.adapter_name,
                "check_id": outcome.check_id,
                "compliant": outcome.compliant,
                "score": outcome.score,
            },
        )

    def store_violation(self, violation: Any) -> None:
        """Store a compliance violation for KM sync.

        Args:
            violation: A Violation or ViolationOutcome.
        """
        if isinstance(violation, ViolationOutcome):
            outcome = violation
        else:
            outcome = ViolationOutcome.from_violation(violation)

        outcome.metadata["km_sync_pending"] = True
        outcome.metadata["km_sync_requested_at"] = datetime.now(timezone.utc).isoformat()
        self._pending_violations.append(outcome)

        self._emit_event(
            "km_adapter_forward_sync",
            {
                "adapter": self.adapter_name,
                "violation_id": outcome.violation_id,
                "severity": outcome.severity,
                "framework": outcome.framework_id,
            },
        )

    def get(self, record_id: str) -> CheckOutcome | ViolationOutcome | None:
        """Get a record by ID (check or violation)."""
        clean_id = record_id
        if record_id.startswith("cc_"):
            clean_id = record_id[3:]
            return self._synced_checks.get(clean_id)
        elif record_id.startswith("cv_"):
            clean_id = record_id[3:]
            return self._synced_violations.get(clean_id)
        # Try both
        return self._synced_checks.get(clean_id) or self._synced_violations.get(clean_id)

    async def get_async(self, record_id: str) -> CheckOutcome | ViolationOutcome | None:
        return self.get(record_id)

    async def search_violations(
        self,
        framework: str = "",
        severity: str = "",
        status: str = "",
        limit: int = 20,
    ) -> list[ComplianceSearchResult]:
        """Search violations by framework, severity, or status.

        Args:
            framework: Filter by framework ID
            severity: Filter by severity (critical, high, medium, low)
            status: Filter by status (open, investigating, resolved)
            limit: Max results

        Returns:
            List of ComplianceSearchResult.
        """
        results: list[ComplianceSearchResult] = []
        all_violations = list(self._synced_violations.values()) + self._pending_violations

        for v in all_violations:
            if framework and v.framework_id != framework:
                continue
            if severity and v.severity != severity:
                continue
            if status and v.status != status:
                continue

            results.append(
                ComplianceSearchResult(
                    record_id=v.violation_id,
                    record_type="violation",
                    framework=v.framework_id,
                    score=0.0,  # Violations don't have a score
                    severity=v.severity,
                    status=v.status,
                    similarity=1.0,
                    description=v.description[:200],
                )
            )

        return results[:limit]

    async def search_checks(
        self,
        framework: str = "",
        min_score: float = 0.0,
        limit: int = 20,
    ) -> list[ComplianceSearchResult]:
        """Search compliance checks by framework and score."""
        results: list[ComplianceSearchResult] = []
        all_checks = list(self._synced_checks.values()) + self._pending_checks

        for check in all_checks:
            if framework and framework not in check.frameworks_checked:
                continue
            if check.score < min_score:
                continue

            results.append(
                ComplianceSearchResult(
                    record_id=check.check_id,
                    record_type="check",
                    framework=",".join(check.frameworks_checked),
                    score=check.score,
                    status="compliant" if check.compliant else "non-compliant",
                    similarity=1.0,
                    description=f"{check.issue_count} issues ({check.critical_count} critical)",
                )
            )

        return results[:limit]

    def check_to_knowledge_item(self, check: CheckOutcome) -> KnowledgeItem:
        """Convert a CheckOutcome to a KnowledgeItem."""
        from aragora.knowledge.mound.types import KnowledgeItem, KnowledgeSource
        from aragora.knowledge.unified.types import ConfidenceLevel

        status = "compliant" if check.compliant else "non-compliant"
        content = (
            f"Compliance check ({','.join(check.frameworks_checked)}): {status} "
            f"(score: {check.score:.2f}, {check.issue_count} issues)"
        )
        if check.critical_count:
            content += f"\nCritical issues: {check.critical_count}"

        return KnowledgeItem(
            id=f"cc_{check.check_id}",
            content=content,
            source=KnowledgeSource.COMPLIANCE,
            source_id=check.check_id,
            confidence=ConfidenceLevel.from_float(check.score),
            created_at=check.created_at,
            updated_at=check.created_at,
            metadata={
                "record_type": "check",
                "compliant": check.compliant,
                "score": check.score,
                "frameworks_checked": check.frameworks_checked,
                "issue_count": check.issue_count,
                "critical_count": check.critical_count,
                "high_count": check.high_count,
            },
        )

    def violation_to_knowledge_item(self, violation: ViolationOutcome) -> KnowledgeItem:
        """Convert a ViolationOutcome to a KnowledgeItem."""
        from aragora.knowledge.mound.types import KnowledgeItem, KnowledgeSource
        from aragora.knowledge.unified.types import ConfidenceLevel

        content = (
            f"Violation [{violation.severity}] {violation.rule_name}: {violation.description[:300]}"
        )
        if violation.resolution_notes:
            content += f"\nResolution: {violation.resolution_notes[:200]}"

        # Map severity to confidence (higher severity = lower confidence in compliance)
        severity_confidence = {
            "critical": 0.1,
            "high": 0.3,
            "medium": 0.5,
            "low": 0.7,
        }
        confidence = severity_confidence.get(violation.severity.lower(), 0.5)

        return KnowledgeItem(
            id=f"cv_{violation.violation_id}",
            content=content,
            source=KnowledgeSource.COMPLIANCE,
            source_id=violation.violation_id,
            confidence=ConfidenceLevel.from_float(confidence),
            created_at=violation.detected_at,
            updated_at=violation.resolved_at or violation.detected_at,
            metadata={
                "record_type": "violation",
                "policy_id": violation.policy_id,
                "rule_id": violation.rule_id,
                "rule_name": violation.rule_name,
                "framework_id": violation.framework_id,
                "severity": violation.severity,
                "status": violation.status,
                "workspace_id": violation.workspace_id,
                "source": violation.source,
                "resolved": violation.resolved_at is not None,
            },
        )

    def to_knowledge_item(self, record: Any) -> KnowledgeItem:
        """Convert any record to a KnowledgeItem (dispatch by type)."""
        if isinstance(record, CheckOutcome):
            return self.check_to_knowledge_item(record)
        elif isinstance(record, ViolationOutcome):
            return self.violation_to_knowledge_item(record)
        raise TypeError(f"Cannot convert {type(record).__name__} to KnowledgeItem")

    async def sync_to_km(
        self,
        mound: Any,
        min_confidence: float = 0.0,
        batch_size: int = 50,
    ) -> SyncResult:
        """Sync pending checks and violations to Knowledge Mound."""
        start = datetime.now(timezone.utc)
        synced = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        # Sync checks
        for check in self._pending_checks[:batch_size]:
            try:
                km_item = self.check_to_knowledge_item(check)
                if hasattr(mound, "store_item"):
                    await mound.store_item(km_item)
                elif hasattr(mound, "store"):
                    await mound.store(km_item)
                elif hasattr(mound, "_semantic_store"):
                    await mound._semantic_store.store(km_item)

                check.metadata["km_sync_pending"] = False
                check.metadata["km_synced_at"] = datetime.now(timezone.utc).isoformat()
                self._synced_checks[check.check_id] = check
                synced += 1
            except Exception as e:
                failed += 1
                errors.append(f"Failed to sync check {check.check_id}: {e}")
                logger.warning("Failed to sync check %s: %s", check.check_id, e)

        # Sync violations
        remaining_batch = batch_size - len(self._pending_checks[:batch_size])
        for violation in self._pending_violations[: max(remaining_batch, 0)]:
            try:
                km_item = self.violation_to_knowledge_item(violation)
                if hasattr(mound, "store_item"):
                    await mound.store_item(km_item)
                elif hasattr(mound, "store"):
                    await mound.store(km_item)
                elif hasattr(mound, "_semantic_store"):
                    await mound._semantic_store.store(km_item)

                violation.metadata["km_sync_pending"] = False
                violation.metadata["km_synced_at"] = datetime.now(timezone.utc).isoformat()
                self._synced_violations[violation.violation_id] = violation
                synced += 1
            except Exception as e:
                failed += 1
                errors.append(f"Failed to sync violation {violation.violation_id}: {e}")
                logger.warning("Failed to sync violation %s: %s", violation.violation_id, e)

        # Remove synced from pending
        self._pending_checks = [
            c for c in self._pending_checks if c.metadata.get("km_sync_pending") is not False
        ]
        self._pending_violations = [
            v for v in self._pending_violations if v.metadata.get("km_sync_pending") is not False
        ]

        duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return SyncResult(
            records_synced=synced,
            records_skipped=skipped,
            records_failed=failed,
            errors=errors,
            duration_ms=duration_ms,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored compliance data."""
        all_checks = list(self._synced_checks.values())
        all_violations = list(self._synced_violations.values())
        return {
            "total_checks_synced": len(self._synced_checks),
            "total_violations_synced": len(self._synced_violations),
            "pending_checks": len(self._pending_checks),
            "pending_violations": len(self._pending_violations),
            "avg_compliance_score": (
                sum(c.score for c in all_checks) / len(all_checks) if all_checks else 0.0
            ),
            "open_violations": sum(1 for v in all_violations if v.status == "open"),
            "resolved_violations": sum(1 for v in all_violations if v.resolved_at is not None),
            "critical_violations": sum(1 for v in all_violations if v.severity == "critical"),
        }

    # --- SemanticSearchMixin required methods ---

    def _get_record_by_id(self, record_id: str) -> Any | None:
        return self.get(record_id)

    def _record_to_dict(self, record: Any, similarity: float = 0.0) -> dict[str, Any]:
        if isinstance(record, CheckOutcome):
            return {
                "id": record.check_id,
                "type": "check",
                "compliant": record.compliant,
                "score": record.score,
                "frameworks": record.frameworks_checked,
                "similarity": similarity,
            }
        elif isinstance(record, ViolationOutcome):
            return {
                "id": record.violation_id,
                "type": "violation",
                "severity": record.severity,
                "status": record.status,
                "framework": record.framework_id,
                "description": record.description[:200],
                "similarity": similarity,
            }
        return {"id": str(record), "similarity": similarity}

    # --- ReverseFlowMixin required methods ---

    def _get_record_for_validation(self, source_id: str) -> Any | None:
        return self.get(source_id)

    def _apply_km_validation(
        self,
        record: Any,
        km_confidence: float,
        cross_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        record.metadata["km_validated"] = True
        record.metadata["km_validation_confidence"] = km_confidence
        record.metadata["km_validation_timestamp"] = datetime.now(timezone.utc).isoformat()
        if cross_refs:
            record.metadata["km_cross_references"] = cross_refs
        return True

    def _extract_source_id(self, item: dict[str, Any]) -> str | None:
        source_id = item.get("source_id", "")
        for prefix in ("cc_", "cv_"):
            if source_id.startswith(prefix):
                return source_id[3:]
        return source_id or None

    # --- FusionMixin required methods ---

    def _get_fusion_sources(self) -> list[str]:
        return ["debate", "workflow"]

    def _extract_fusible_data(self, km_item: dict[str, Any]) -> dict[str, Any] | None:
        if km_item.get("source") == "compliance":
            return {
                "score": km_item.get("confidence", 0.0),
                "record_type": km_item.get("metadata", {}).get("record_type", ""),
            }
        return None

    def _apply_fusion_result(
        self,
        record: Any,
        fusion_result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        record.metadata["fusion_applied"] = True
        record.metadata["fusion_timestamp"] = datetime.now(timezone.utc).isoformat()
        return True
