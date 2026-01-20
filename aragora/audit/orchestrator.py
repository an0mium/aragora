"""
Multi-Vertical Audit Orchestrator.

Coordinates multiple auditors across enterprise verticals using
hive-mind patterns for comprehensive document analysis.

Features:
- Parallel auditor execution
- Finding aggregation and deduplication
- Cross-auditor correlation
- Priority-based scheduling
- Workspace isolation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Type

from aragora.audit.base_auditor import (
    AuditContext,
    BaseAuditor,
    ChunkData,
)
from aragora.audit.document_auditor import (
    AuditFinding,
    AuditSession,
    FindingSeverity,
)
from aragora.audit.audit_types import (
    SecurityAuditor,
    ComplianceAuditor,
    ConsistencyAuditor,
    QualityAuditor,
    LegalAuditor,
    AccountingAuditor,
    SoftwareAuditor,
    HealthcareAuditor,
    RegulatoryAuditor,
    AcademicAuditor,
)

logger = logging.getLogger(__name__)


class AuditVertical(str, Enum):
    """Enterprise audit verticals."""

    # Core verticals
    SECURITY = "security"
    COMPLIANCE = "compliance"
    QUALITY = "quality"
    CONSISTENCY = "consistency"

    # Domain verticals
    LEGAL = "legal"
    ACCOUNTING = "accounting"
    SOFTWARE = "software"
    HEALTHCARE = "healthcare"
    REGULATORY = "regulatory"
    ACADEMIC = "academic"


# Mapping of verticals to auditor classes
VERTICAL_AUDITORS: Dict[AuditVertical, Type[BaseAuditor]] = {
    AuditVertical.SECURITY: SecurityAuditor,
    AuditVertical.COMPLIANCE: ComplianceAuditor,
    AuditVertical.QUALITY: QualityAuditor,
    AuditVertical.CONSISTENCY: ConsistencyAuditor,
    AuditVertical.LEGAL: LegalAuditor,
    AuditVertical.ACCOUNTING: AccountingAuditor,
    AuditVertical.SOFTWARE: SoftwareAuditor,
    AuditVertical.HEALTHCARE: HealthcareAuditor,
    AuditVertical.REGULATORY: RegulatoryAuditor,
    AuditVertical.ACADEMIC: AcademicAuditor,
}


@dataclass
class AuditProfile:
    """
    Pre-configured audit profile for specific use cases.

    Combines multiple verticals with appropriate settings.
    """

    name: str
    description: str
    verticals: List[AuditVertical]
    priority_order: List[AuditVertical]  # Order for serial execution
    parallel_execution: bool = True
    max_concurrent: int = 5
    confidence_threshold: float = 0.5
    include_low_severity: bool = False
    custom_config: Dict[str, Any] = field(default_factory=dict)


# Pre-defined audit profiles
AUDIT_PROFILES = {
    "enterprise_full": AuditProfile(
        name="Enterprise Full Audit",
        description="Comprehensive audit across all enterprise verticals",
        verticals=[v for v in AuditVertical],
        priority_order=[
            AuditVertical.SECURITY,
            AuditVertical.COMPLIANCE,
            AuditVertical.HEALTHCARE,
            AuditVertical.REGULATORY,
            AuditVertical.LEGAL,
            AuditVertical.ACCOUNTING,
            AuditVertical.SOFTWARE,
            AuditVertical.QUALITY,
            AuditVertical.CONSISTENCY,
            AuditVertical.ACADEMIC,
        ],
        parallel_execution=True,
        max_concurrent=6,
    ),
    "healthcare_hipaa": AuditProfile(
        name="Healthcare HIPAA Audit",
        description="HIPAA-focused audit for healthcare documents",
        verticals=[
            AuditVertical.HEALTHCARE,
            AuditVertical.SECURITY,
            AuditVertical.COMPLIANCE,
            AuditVertical.QUALITY,
        ],
        priority_order=[
            AuditVertical.HEALTHCARE,
            AuditVertical.SECURITY,
            AuditVertical.COMPLIANCE,
            AuditVertical.QUALITY,
        ],
        confidence_threshold=0.6,
        custom_config={"hipaa_strict": True},
    ),
    "financial_sox": AuditProfile(
        name="Financial SOX Audit",
        description="SOX-focused audit for financial documents",
        verticals=[
            AuditVertical.ACCOUNTING,
            AuditVertical.REGULATORY,
            AuditVertical.COMPLIANCE,
            AuditVertical.SECURITY,
        ],
        priority_order=[
            AuditVertical.ACCOUNTING,
            AuditVertical.REGULATORY,
            AuditVertical.COMPLIANCE,
            AuditVertical.SECURITY,
        ],
        confidence_threshold=0.7,
    ),
    "legal_contract": AuditProfile(
        name="Legal Contract Audit",
        description="Contract review and legal compliance audit",
        verticals=[
            AuditVertical.LEGAL,
            AuditVertical.COMPLIANCE,
            AuditVertical.REGULATORY,
            AuditVertical.CONSISTENCY,
        ],
        priority_order=[
            AuditVertical.LEGAL,
            AuditVertical.COMPLIANCE,
            AuditVertical.REGULATORY,
            AuditVertical.CONSISTENCY,
        ],
    ),
    "code_security": AuditProfile(
        name="Code Security Audit",
        description="Security-focused code and software audit",
        verticals=[
            AuditVertical.SOFTWARE,
            AuditVertical.SECURITY,
            AuditVertical.COMPLIANCE,
            AuditVertical.QUALITY,
        ],
        priority_order=[
            AuditVertical.SECURITY,
            AuditVertical.SOFTWARE,
            AuditVertical.COMPLIANCE,
            AuditVertical.QUALITY,
        ],
        include_low_severity=True,  # Include all security findings
    ),
    "academic_research": AuditProfile(
        name="Academic Research Audit",
        description="Citation verification and plagiarism detection",
        verticals=[
            AuditVertical.ACADEMIC,
            AuditVertical.QUALITY,
            AuditVertical.CONSISTENCY,
        ],
        priority_order=[
            AuditVertical.ACADEMIC,
            AuditVertical.QUALITY,
            AuditVertical.CONSISTENCY,
        ],
        include_low_severity=True,
    ),
    "quick_security": AuditProfile(
        name="Quick Security Scan",
        description="Fast security-only scan",
        verticals=[AuditVertical.SECURITY],
        priority_order=[AuditVertical.SECURITY],
        parallel_execution=False,
    ),
}


@dataclass
class OrchestratorResult:
    """Result from multi-vertical audit orchestration."""

    session_id: str
    profile: str
    verticals_run: List[str]
    findings: List[AuditFinding]
    findings_by_vertical: Dict[str, List[AuditFinding]]
    findings_by_severity: Dict[str, int]
    total_chunks_processed: int
    duration_ms: float
    errors: List[str]
    started_at: datetime
    completed_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "profile": self.profile,
            "verticals_run": self.verticals_run,
            "findings_count": len(self.findings),
            "findings_by_vertical": {k: len(v) for k, v in self.findings_by_vertical.items()},
            "findings_by_severity": self.findings_by_severity,
            "total_chunks_processed": self.total_chunks_processed,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
        }


class AuditOrchestrator:
    """
    Orchestrates multi-vertical audits with hive-mind coordination.

    Manages parallel auditor execution, finding aggregation, and
    cross-auditor correlation for comprehensive document analysis.
    """

    def __init__(
        self,
        profile: Optional[str] = None,
        verticals: Optional[List[AuditVertical]] = None,
        workspace_id: Optional[str] = None,
        max_concurrent: int = 5,
    ):
        """
        Initialize audit orchestrator.

        Args:
            profile: Pre-defined audit profile name
            verticals: Custom list of verticals (overrides profile)
            workspace_id: Workspace for isolation
            max_concurrent: Maximum concurrent auditors
        """
        self._workspace_id = workspace_id
        self._max_concurrent = max_concurrent

        # Load profile or use custom verticals
        if profile and profile in AUDIT_PROFILES:
            self._profile = AUDIT_PROFILES[profile]
        elif verticals:
            self._profile = AuditProfile(
                name="Custom",
                description="Custom audit configuration",
                verticals=verticals,
                priority_order=verticals,
            )
        else:
            self._profile = AUDIT_PROFILES["enterprise_full"]

        # Initialize auditors
        self._auditors: Dict[AuditVertical, BaseAuditor] = {}
        self._initialize_auditors()

        # Runtime state
        self._findings: List[AuditFinding] = []
        self._errors: List[str] = []

    def _initialize_auditors(self) -> None:
        """Initialize auditor instances for selected verticals."""
        for vertical in self._profile.verticals:
            auditor_class = VERTICAL_AUDITORS.get(vertical)
            if auditor_class:
                try:
                    self._auditors[vertical] = auditor_class()
                except Exception as e:
                    logger.warning(f"Failed to initialize {vertical.value} auditor: {e}")

    async def run(
        self,
        chunks: Sequence[Dict[str, Any]],
        session: AuditSession,
    ) -> OrchestratorResult:
        """
        Run multi-vertical audit.

        Args:
            chunks: Document chunks to audit
            session: Audit session context

        Returns:
            Orchestration result with aggregated findings
        """
        started_at = datetime.now(timezone.utc)
        self._findings = []
        self._errors = []

        # Convert chunks to ChunkData
        chunk_data = [ChunkData.from_dict(c) for c in chunks]

        # Create audit context
        context = AuditContext(
            session=session,
            workspace_id=self._workspace_id,
            confidence_threshold=self._profile.confidence_threshold,
            include_low_severity=self._profile.include_low_severity,
            custom_params=self._profile.custom_config,
        )

        # Run auditors
        if self._profile.parallel_execution:
            await self._run_parallel(chunk_data, context)
        else:
            await self._run_serial(chunk_data, context)

        # Post-processing
        self._findings = self._deduplicate_findings(self._findings)
        self._findings = self._correlate_findings(self._findings)

        completed_at = datetime.now(timezone.utc)
        duration_ms = (completed_at - started_at).total_seconds() * 1000

        # Aggregate by vertical and severity
        findings_by_vertical: Dict[str, List[AuditFinding]] = {}
        findings_by_severity: Dict[str, int] = {s.value: 0 for s in FindingSeverity}

        for finding in self._findings:
            # By vertical (use category prefix or default)
            vertical = finding.category.split("_")[0] if "_" in finding.category else "other"
            if vertical not in findings_by_vertical:
                findings_by_vertical[vertical] = []
            findings_by_vertical[vertical].append(finding)

            # By severity
            findings_by_severity[finding.severity.value] += 1

        return OrchestratorResult(
            session_id=session.id,
            profile=self._profile.name,
            verticals_run=[v.value for v in self._auditors.keys()],
            findings=self._findings,
            findings_by_vertical=findings_by_vertical,
            findings_by_severity=findings_by_severity,
            total_chunks_processed=len(chunk_data),
            duration_ms=duration_ms,
            errors=self._errors,
            started_at=started_at,
            completed_at=completed_at,
        )

    async def _run_parallel(
        self,
        chunks: Sequence[ChunkData],
        context: AuditContext,
    ) -> None:
        """Run auditors in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def run_auditor(vertical: AuditVertical, auditor: BaseAuditor) -> None:
            async with semaphore:
                try:
                    logger.debug(f"Starting {vertical.value} auditor")

                    # Pre-audit hook
                    await auditor.pre_audit_hook(context)

                    # Analyze chunks
                    auditor_findings = []
                    for chunk in chunks:
                        try:
                            chunk_findings = await auditor.analyze_chunk(chunk, context)
                            for finding in chunk_findings:
                                if auditor.validate_finding(finding, context):
                                    auditor_findings.append(finding)
                        except Exception as e:
                            logger.warning(f"Error in {vertical.value} on chunk {chunk.id}: {e}")

                    # Cross-document analysis
                    if auditor.capabilities.supports_cross_document:
                        try:
                            cross_findings = await auditor.cross_document_analysis(chunks, context)
                            for finding in cross_findings:
                                if auditor.validate_finding(finding, context):
                                    auditor_findings.append(finding)
                        except Exception as e:
                            logger.warning(f"Cross-doc analysis error in {vertical.value}: {e}")

                    # Post-audit hook
                    auditor_findings = await auditor.post_audit_hook(auditor_findings, context)

                    self._findings.extend(auditor_findings)
                    logger.debug(
                        f"Completed {vertical.value} with {len(auditor_findings)} findings"
                    )

                except Exception as e:
                    error_msg = f"Auditor {vertical.value} failed: {e}"
                    logger.error(error_msg)
                    self._errors.append(error_msg)

        # Run all auditors concurrently
        tasks = [run_auditor(vertical, auditor) for vertical, auditor in self._auditors.items()]
        await asyncio.gather(*tasks)

    async def _run_serial(
        self,
        chunks: Sequence[ChunkData],
        context: AuditContext,
    ) -> None:
        """Run auditors serially in priority order."""
        for vertical in self._profile.priority_order:
            auditor = self._auditors.get(vertical)
            if not auditor:
                continue

            try:
                logger.debug(f"Running {vertical.value} auditor")

                await auditor.pre_audit_hook(context)

                auditor_findings = []
                for chunk in chunks:
                    chunk_findings = await auditor.analyze_chunk(chunk, context)
                    for finding in chunk_findings:
                        if auditor.validate_finding(finding, context):
                            auditor_findings.append(finding)

                if auditor.capabilities.supports_cross_document:
                    cross_findings = await auditor.cross_document_analysis(chunks, context)
                    for finding in cross_findings:
                        if auditor.validate_finding(finding, context):
                            auditor_findings.append(finding)

                auditor_findings = await auditor.post_audit_hook(auditor_findings, context)
                self._findings.extend(auditor_findings)

            except Exception as e:
                error_msg = f"Auditor {vertical.value} failed: {e}"
                logger.error(error_msg)
                self._errors.append(error_msg)

    def _deduplicate_findings(
        self,
        findings: List[AuditFinding],
    ) -> List[AuditFinding]:
        """Remove duplicate findings across auditors."""
        seen: Set[tuple] = set()
        unique = []

        for finding in findings:
            # Create dedup key from core attributes
            key = (
                finding.document_id,
                finding.title,
                finding.category,
                finding.severity.value,
                finding.evidence_text[:100] if finding.evidence_text else "",
            )

            if key not in seen:
                seen.add(key)
                unique.append(finding)

        return unique

    def _correlate_findings(
        self,
        findings: List[AuditFinding],
    ) -> List[AuditFinding]:
        """
        Correlate findings across auditors to identify patterns.

        Enhances findings with cross-auditor insights.
        """
        # Group by document
        by_document: Dict[str, List[AuditFinding]] = {}
        for finding in findings:
            if finding.document_id not in by_document:
                by_document[finding.document_id] = []
            by_document[finding.document_id].append(finding)

        # Check for correlated issues (multiple auditors flagging same area)
        for doc_id, doc_findings in by_document.items():
            if len(doc_findings) > 10:
                # Multiple issues in one document - flag as high-risk
                for finding in doc_findings:
                    if finding.severity == FindingSeverity.MEDIUM:
                        finding.tags.append("multi_auditor_flagged")

        return findings

    @classmethod
    def list_profiles(cls) -> List[Dict[str, Any]]:
        """List available audit profiles."""
        return [
            {
                "name": name,
                "description": profile.description,
                "verticals": [v.value for v in profile.verticals],
            }
            for name, profile in AUDIT_PROFILES.items()
        ]

    @classmethod
    def list_verticals(cls) -> List[Dict[str, Any]]:
        """List available audit verticals."""
        return [
            {
                "id": v.value,
                "name": v.name.replace("_", " ").title(),
                "auditor": VERTICAL_AUDITORS[v].__name__,
            }
            for v in AuditVertical
        ]


__all__ = [
    "AuditOrchestrator",
    "AuditVertical",
    "AuditProfile",
    "OrchestratorResult",
    "AUDIT_PROFILES",
    "VERTICAL_AUDITORS",
]
