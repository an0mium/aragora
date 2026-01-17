"""Codebase Auditor for Nomic Loop integration.

Adapts the document auditing tools for codebase analysis, enabling:
- Pre-cycle auditing to identify improvement opportunities
- Verification of changes for consistency
- Documentation drift detection
- Security and compliance scanning of code
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from aragora.audit.document_auditor import (
    AuditConfig,
    AuditFinding,
    AuditSession,
    AuditType,
    DocumentAuditor,
    FindingSeverity,
)
from aragora.audit.audit_types.consistency import ConsistencyAuditor
from aragora.documents.chunking.token_counter import TokenCounter
from aragora.documents.chunking.strategies import (
    ChunkingConfig,
    SemanticChunking,
    get_chunking_strategy,
)

logger = logging.getLogger(__name__)


@dataclass
class CodebaseAuditConfig:
    """Configuration for codebase auditing."""

    # Paths to audit
    include_paths: list[str] = field(default_factory=lambda: ["aragora/", "scripts/", "docs/"])
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["__pycache__", ".git", "node_modules", ".venv", "*.pyc"]
    )

    # File types to audit
    code_extensions: list[str] = field(default_factory=lambda: [".py", ".ts", ".tsx", ".js"])
    doc_extensions: list[str] = field(default_factory=lambda: [".md", ".rst", ".txt"])

    # Audit types to run
    audit_types: list[AuditType] = field(
        default_factory=lambda: [AuditType.CONSISTENCY, AuditType.QUALITY, AuditType.SECURITY]
    )

    # Token management
    max_context_tokens: int = 500_000
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Filtering
    min_severity: FindingSeverity = FindingSeverity.MEDIUM
    min_confidence: float = 0.7
    max_findings_per_cycle: int = 10

    # Performance
    max_concurrent_files: int = 5
    timeout_per_file: float = 30.0


@dataclass
class ImprovementProposal:
    """A structured improvement proposal derived from audit findings."""

    id: str
    title: str
    description: str
    finding_ids: list[str]  # Source audit findings
    severity: FindingSeverity
    confidence: float
    estimated_effort: str = "medium"  # low, medium, high
    affected_files: list[str] = field(default_factory=list)
    suggested_fix: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "finding_ids": self.finding_ids,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "estimated_effort": self.estimated_effort,
            "affected_files": self.affected_files,
            "suggested_fix": self.suggested_fix,
            "tags": self.tags,
        }


@dataclass
class CodebaseAuditResult:
    """Result of a codebase audit."""

    session_id: str
    started_at: datetime
    completed_at: datetime
    files_audited: int
    total_tokens: int
    findings: list[AuditFinding]
    proposals: list[ImprovementProposal]
    summary: str = ""

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def findings_by_severity(self) -> dict[str, int]:
        counts = {}
        for finding in self.findings:
            sev = finding.severity.value
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "duration_seconds": self.duration_seconds,
            "files_audited": self.files_audited,
            "total_tokens": self.total_tokens,
            "findings_count": len(self.findings),
            "findings_by_severity": self.findings_by_severity,
            "proposals_count": len(self.proposals),
            "summary": self.summary,
        }


class CodebaseAuditor:
    """Audits codebase for improvement opportunities.

    Adapts DocumentAuditor for code-specific analysis:
    - Treats Python files as documents
    - Compares code vs documentation
    - Detects code-doc drift
    - Finds security issues in code
    - Identifies quality issues

    Integration with Nomic Loop:
    - Called during Context phase to find improvement candidates
    - Called during Verify phase to validate changes
    """

    def __init__(
        self,
        root_path: Path,
        config: Optional[CodebaseAuditConfig] = None,
        document_auditor: Optional[DocumentAuditor] = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        """Initialize the codebase auditor.

        Args:
            root_path: Root path of the codebase to audit
            config: Audit configuration
            document_auditor: Optional DocumentAuditor instance
            token_counter: Optional TokenCounter instance
        """
        self.root_path = Path(root_path)
        self.config = config or CodebaseAuditConfig()
        self.token_counter = token_counter or TokenCounter()

        # Initialize document auditor
        if document_auditor:
            self.document_auditor = document_auditor
        else:
            self.document_auditor = DocumentAuditor(
                config=AuditConfig(
                    min_confidence=self.config.min_confidence,
                )
            )

        # Initialize consistency auditor for cross-file checks
        self.consistency_auditor = ConsistencyAuditor()

        # Chunking strategy
        self._chunker = None

    def _get_chunker(self) -> SemanticChunking:
        """Get or create chunking strategy."""
        if self._chunker is None:
            chunk_config = ChunkingConfig(
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            self._chunker = get_chunking_strategy("semantic", chunk_config)
        return self._chunker

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if a file should be included in the audit."""
        # Check exclude patterns
        path_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            if pattern in path_str:
                return False

        # Check file extension
        ext = file_path.suffix.lower()
        return ext in self.config.code_extensions or ext in self.config.doc_extensions

    def _collect_files(self) -> list[Path]:
        """Collect files to audit."""
        files = []

        for include_path in self.config.include_paths:
            base = self.root_path / include_path
            if not base.exists():
                continue

            if base.is_file():
                if self._should_include_file(base):
                    files.append(base)
            else:
                for file_path in base.rglob("*"):
                    if file_path.is_file() and self._should_include_file(file_path):
                        files.append(file_path)

        return files

    async def audit_codebase(self) -> CodebaseAuditResult:
        """Run a full codebase audit.

        Returns:
            CodebaseAuditResult with findings and proposals
        """
        start_time = datetime.utcnow()
        session_id = f"codebase_audit_{start_time.strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"[{session_id}] Starting codebase audit")

        # Collect files
        files = self._collect_files()
        logger.info(f"[{session_id}] Found {len(files)} files to audit")

        # Process files and collect chunks
        all_chunks = []
        total_tokens = 0
        chunker = self._get_chunker()

        for file_path in files:
            try:
                content = file_path.read_text()
                tokens = self.token_counter.count(content)
                total_tokens += tokens

                # Chunk the file
                rel_path = str(file_path.relative_to(self.root_path))
                chunks = chunker.chunk(content, document_id=rel_path)

                for chunk in chunks:
                    all_chunks.append(
                        {
                            "id": f"{rel_path}:{chunk.sequence}",
                            "document_id": rel_path,
                            "content": chunk.content,
                            "file_type": file_path.suffix,
                        }
                    )

            except Exception as e:
                logger.warning(f"[{session_id}] Failed to process {file_path}: {e}")

        logger.info(f"[{session_id}] Created {len(all_chunks)} chunks, {total_tokens:,} tokens")

        # Run audits
        findings = []

        # Consistency audit
        if AuditType.CONSISTENCY in self.config.audit_types:
            consistency_findings = await self._run_consistency_audit(all_chunks, session_id)
            findings.extend(consistency_findings)

        # Pattern-based audits (security, quality)
        pattern_findings = await self._run_pattern_audits(all_chunks, session_id)
        findings.extend(pattern_findings)

        # Filter by severity and confidence
        findings = [
            f
            for f in findings
            if f.confidence >= self.config.min_confidence
            and self._severity_at_least(f.severity, self.config.min_severity)
        ]

        # Sort by severity and confidence
        findings.sort(key=lambda f: (self._severity_rank(f.severity), f.confidence), reverse=True)

        # Limit findings
        findings = findings[: self.config.max_findings_per_cycle * 2]

        # Generate improvement proposals
        proposals = self.findings_to_proposals(findings)

        end_time = datetime.utcnow()

        # Build summary
        summary = self._build_summary(findings, proposals)

        result = CodebaseAuditResult(
            session_id=session_id,
            started_at=start_time,
            completed_at=end_time,
            files_audited=len(files),
            total_tokens=total_tokens,
            findings=findings,
            proposals=proposals,
            summary=summary,
        )

        logger.info(
            f"[{session_id}] Audit complete: {len(findings)} findings, "
            f"{len(proposals)} proposals in {result.duration_seconds:.1f}s"
        )

        return result

    async def _run_consistency_audit(
        self, chunks: list[dict], session_id: str
    ) -> list[AuditFinding]:
        """Run consistency audit on chunks."""
        try:
            mock_session = AuditSession(
                id=session_id,
                document_ids=[c["document_id"] for c in chunks[:10]],
                audit_types=[AuditType.CONSISTENCY],
            )
            findings = await self.consistency_auditor.audit(chunks, mock_session)
            return findings
        except Exception as e:
            logger.warning(f"Consistency audit failed: {e}")
            return []

    async def _run_pattern_audits(self, chunks: list[dict], session_id: str) -> list[AuditFinding]:
        """Run pattern-based audits (security, quality)."""
        findings = []

        for chunk in chunks:
            content = chunk.get("content", "")
            doc_id = chunk.get("document_id", "unknown")
            file_type = chunk.get("file_type", "")

            # Security patterns for code files
            if file_type in [".py", ".ts", ".js", ".tsx"]:
                security_findings = self._check_security_patterns(content, doc_id, chunk["id"])
                findings.extend(security_findings)

            # Quality patterns
            quality_findings = self._check_quality_patterns(content, doc_id, chunk["id"])
            findings.extend(quality_findings)

        return findings

    def _check_security_patterns(
        self, content: str, doc_id: str, chunk_id: str
    ) -> list[AuditFinding]:
        """Check for security issues in code."""
        import re

        findings = []

        # Pattern: Hardcoded secrets
        secret_patterns = [
            (
                r'(?:api[_-]?key|apikey|secret|password|token)\s*[=:]\s*["\'][^"\']{8,}["\']',
                "Potential hardcoded secret",
            ),
            (r"(?:AWS|AKIA)[A-Z0-9]{16,}", "Potential AWS key"),
            (r"sk-[a-zA-Z0-9]{20,}", "Potential OpenAI API key"),
        ]

        for pattern, description in secret_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append(
                    AuditFinding(
                        audit_type=AuditType.SECURITY,
                        category="hardcoded_secret",
                        title=description,
                        description=f"Found in {doc_id}: {match.group()[:50]}...",
                        severity=FindingSeverity.HIGH,
                        confidence=0.8,
                        evidence_text=match.group()[:100],
                        evidence_location=f"{doc_id}:{chunk_id}",
                        session_id="",
                    )
                )

        # Pattern: SQL injection risk
        if re.search(r'execute\s*\(\s*f["\']', content) or re.search(
            r"\.format\s*\([^)]*\)\s*\)", content
        ):
            findings.append(
                AuditFinding(
                    audit_type=AuditType.SECURITY,
                    category="sql_injection",
                    title="Potential SQL injection vulnerability",
                    description=f"String interpolation in database query in {doc_id}",
                    severity=FindingSeverity.HIGH,
                    confidence=0.7,
                    evidence_location=f"{doc_id}:{chunk_id}",
                    session_id="",
                )
            )

        return findings

    def _check_quality_patterns(
        self, content: str, doc_id: str, chunk_id: str
    ) -> list[AuditFinding]:
        """Check for quality issues."""
        import re

        findings = []

        # Pattern: TODO/FIXME comments
        todo_matches = list(
            re.finditer(r"#\s*(TODO|FIXME|XXX|HACK)[:.]?\s*(.{0,100})", content, re.IGNORECASE)
        )
        if todo_matches:
            for match in todo_matches[:3]:  # Limit per file
                findings.append(
                    AuditFinding(
                        audit_type=AuditType.QUALITY,
                        category="incomplete_code",
                        title=f"{match.group(1).upper()} comment",
                        description=f"{match.group(1)}: {match.group(2)[:80]}",
                        severity=FindingSeverity.LOW,
                        confidence=0.9,
                        evidence_text=match.group(0),
                        evidence_location=f"{doc_id}:{chunk_id}",
                        session_id="",
                    )
                )

        # Pattern: Long functions (rough heuristic)
        if content.count("\n") > 100:
            findings.append(
                AuditFinding(
                    audit_type=AuditType.QUALITY,
                    category="complexity",
                    title="Large code block",
                    description=f"File chunk {doc_id} has {content.count(chr(10))} lines - consider refactoring",
                    severity=FindingSeverity.LOW,
                    confidence=0.6,
                    evidence_location=f"{doc_id}:{chunk_id}",
                    session_id="",
                )
            )

        return findings

    def _severity_rank(self, severity: FindingSeverity) -> int:
        """Get numeric rank for severity (higher = more severe)."""
        ranks = {
            FindingSeverity.CRITICAL: 4,
            FindingSeverity.HIGH: 3,
            FindingSeverity.MEDIUM: 2,
            FindingSeverity.LOW: 1,
        }
        return ranks.get(severity, 0)

    def _severity_at_least(self, severity: FindingSeverity, minimum: FindingSeverity) -> bool:
        """Check if severity is at least the minimum."""
        return self._severity_rank(severity) >= self._severity_rank(minimum)

    def findings_to_proposals(
        self,
        findings: list[AuditFinding],
        max_proposals: int = None,
    ) -> list[ImprovementProposal]:
        """Convert audit findings to structured improvement proposals.

        Groups related findings and creates actionable proposals.
        """
        max_proposals = max_proposals or self.config.max_findings_per_cycle
        proposals = []

        # Group findings by category and location
        grouped: dict[str, list[AuditFinding]] = {}
        for finding in findings:
            key = f"{finding.category}:{finding.audit_type.value}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(finding)

        # Create proposals from groups
        for key, group_findings in grouped.items():
            if not group_findings:
                continue

            # Use highest severity finding as representative
            rep = max(group_findings, key=lambda f: self._severity_rank(f.severity))

            # Collect affected files
            affected_files = list(
                set(
                    f.evidence_location.split(":")[0] for f in group_findings if f.evidence_location
                )
            )

            proposal = ImprovementProposal(
                id=f"proposal_{key}_{len(proposals)}",
                title=f"Fix {rep.category}: {rep.title}",
                description=self._build_proposal_description(group_findings),
                finding_ids=[f.id for f in group_findings if hasattr(f, "id")],
                severity=rep.severity,
                confidence=sum(f.confidence for f in group_findings) / len(group_findings),
                affected_files=affected_files[:5],
                suggested_fix=self._suggest_fix(rep),
                tags=[rep.audit_type.value, rep.category],
            )
            proposals.append(proposal)

        # Sort by severity and limit
        proposals.sort(key=lambda p: self._severity_rank(p.severity), reverse=True)
        return proposals[:max_proposals]

    def _build_proposal_description(self, findings: list[AuditFinding]) -> str:
        """Build a description for a proposal from its findings."""
        if len(findings) == 1:
            return findings[0].description
        return f"{findings[0].description} (and {len(findings) - 1} similar issues)"

    def _suggest_fix(self, finding: AuditFinding) -> str:
        """Suggest a fix for a finding."""
        suggestions = {
            "hardcoded_secret": "Move secrets to environment variables or a secrets manager",
            "sql_injection": "Use parameterized queries instead of string interpolation",
            "incomplete_code": "Complete the TODO item or remove if no longer relevant",
            "complexity": "Consider breaking into smaller, focused functions",
            "date_mismatch": "Update dates to be consistent across documents",
            "definition_conflict": "Reconcile conflicting definitions",
        }
        return suggestions.get(finding.category, "Review and address the finding")

    def _build_summary(
        self, findings: list[AuditFinding], proposals: list[ImprovementProposal]
    ) -> str:
        """Build a human-readable summary of audit results."""
        severity_counts = {}
        for f in findings:
            sev = f.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        parts = [f"Found {len(findings)} issues:"]
        for sev in ["critical", "high", "medium", "low"]:
            if sev in severity_counts:
                parts.append(f"  - {severity_counts[sev]} {sev}")

        parts.append(f"\nGenerated {len(proposals)} improvement proposals")

        return "\n".join(parts)

    async def audit_files(self, files: list[Path]) -> list[AuditFinding]:
        """Audit specific files (for incremental verification).

        Useful for verifying changes don't introduce new issues.
        """
        all_chunks = []
        chunker = self._get_chunker()

        for file_path in files:
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text()
                rel_path = str(file_path.relative_to(self.root_path))
                chunks = chunker.chunk(content, document_id=rel_path)

                for chunk in chunks:
                    all_chunks.append(
                        {
                            "id": f"{rel_path}:{chunk.sequence}",
                            "document_id": rel_path,
                            "content": chunk.content,
                            "file_type": file_path.suffix,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")

        # Run pattern audits only for speed
        findings = await self._run_pattern_audits(all_chunks, "incremental")
        return findings
