"""
Document Auditor Orchestrator.

Coordinates multi-agent document auditing for enterprise use cases:
- Security vulnerability detection
- Compliance verification (GDPR, HIPAA, SOC2)
- Cross-document consistency checking
- Quality and completeness analysis

Uses the Aragora debate framework for multi-agent verification.

Usage:
    from aragora.audit.document_auditor import DocumentAuditor, AuditSession

    auditor = DocumentAuditor()
    session = await auditor.create_session(
        document_ids=["doc1", "doc2"],
        audit_types=["security", "compliance"],
    )
    await auditor.run_audit(session.id)
    findings = await auditor.get_findings(session.id)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class AuditStatus(str, Enum):
    """Status of an audit session."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AuditType(str, Enum):
    """Types of document audits."""

    SECURITY = "security"
    COMPLIANCE = "compliance"
    CONSISTENCY = "consistency"
    QUALITY = "quality"
    ALL = "all"


class FindingSeverity(str, Enum):
    """Severity levels for audit findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingStatus(str, Enum):
    """Status of an audit finding."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    WONT_FIX = "wont_fix"


@dataclass
class AuditFinding:
    """A finding from the document audit."""

    id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    document_id: str = ""
    chunk_id: Optional[str] = None

    # Classification
    audit_type: AuditType = AuditType.QUALITY
    category: str = ""  # Specific category within type
    severity: FindingSeverity = FindingSeverity.MEDIUM
    confidence: float = 0.8

    # Content
    title: str = ""
    description: str = ""
    evidence_text: str = ""
    evidence_location: str = ""  # Page/line reference

    # Remediation
    recommendation: str = ""
    affected_scope: str = ""  # file, section, document, collection

    # Agent attribution
    found_by: str = ""
    confirmed_by: list[str] = field(default_factory=list)
    disputed_by: list[str] = field(default_factory=list)

    # Metadata
    status: FindingStatus = FindingStatus.OPEN
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "audit_type": self.audit_type.value,
            "category": self.category,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
            "evidence_text": self.evidence_text,
            "evidence_location": self.evidence_location,
            "recommendation": self.recommendation,
            "affected_scope": self.affected_scope,
            "found_by": self.found_by,
            "confirmed_by": self.confirmed_by,
            "disputed_by": self.disputed_by,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditFinding":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            session_id=data.get("session_id", ""),
            document_id=data.get("document_id", ""),
            chunk_id=data.get("chunk_id"),
            audit_type=AuditType(data.get("audit_type", "quality")),
            category=data.get("category", ""),
            severity=FindingSeverity(data.get("severity", "medium")),
            confidence=data.get("confidence", 0.8),
            title=data.get("title", ""),
            description=data.get("description", ""),
            evidence_text=data.get("evidence_text", ""),
            evidence_location=data.get("evidence_location", ""),
            recommendation=data.get("recommendation", ""),
            affected_scope=data.get("affected_scope", ""),
            found_by=data.get("found_by", ""),
            confirmed_by=data.get("confirmed_by", []),
            disputed_by=data.get("disputed_by", []),
            status=FindingStatus(data.get("status", "open")),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else datetime.now(timezone.utc)
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if isinstance(data.get("updated_at"), str)
                else datetime.now(timezone.utc)
            ),
            tags=data.get("tags", []),
        )


@dataclass
class AuditSession:
    """An audit session tracking state and progress."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""

    # Documents
    document_ids: list[str] = field(default_factory=list)
    total_chunks: int = 0
    processed_chunks: int = 0

    # Configuration
    audit_types: list[AuditType] = field(default_factory=lambda: [AuditType.ALL])
    model: str = "gemini-3-pro"
    max_tokens_per_call: int = 500000

    # Status
    status: AuditStatus = AuditStatus.PENDING
    progress: float = 0.0
    current_phase: str = ""

    # Results
    findings: list[AuditFinding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metadata
    created_by: str = ""
    org_id: str = ""

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get audit duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    @property
    def findings_by_severity(self) -> dict[str, int]:
        """Count findings by severity."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            sev = finding.severity.value
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "document_ids": self.document_ids,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "audit_types": [t.value for t in self.audit_types],
            "model": self.model,
            "max_tokens_per_call": self.max_tokens_per_call,
            "status": self.status.value,
            "progress": self.progress,
            "current_phase": self.current_phase,
            "findings_count": len(self.findings),
            "findings_by_severity": self.findings_by_severity,
            "errors": self.errors,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "created_by": self.created_by,
            "org_id": self.org_id,
        }


@dataclass
class AuditConfig:
    """Configuration for document auditing."""

    # Model settings
    primary_model: str = "gemini-3-pro"  # Large context for initial scan
    verification_model: str = "claude-3.5-sonnet"  # Deep reasoning for verification
    adversarial_model: str = "gpt-4-turbo"  # Adversarial checking

    # Processing settings
    max_tokens_per_call: int = 500000
    enable_chunking: bool = True
    chunk_overlap: float = 0.1

    # Detection thresholds
    min_confidence: float = 0.7
    require_confirmation: bool = True
    confirmation_threshold: int = 2  # Agents needed to confirm

    # Parallelism
    max_concurrent_documents: int = 5
    max_concurrent_chunks: int = 10

    # Timeouts
    document_timeout_seconds: int = 300
    chunk_timeout_seconds: int = 60

    # Hive-Mind orchestration (for multi-document audits)
    use_hive_mind: bool = True  # Enable Hive-Mind for multi-document audits
    consensus_verification: bool = True  # Use Byzantine consensus for critical findings

    # Knowledge pipeline integration
    use_knowledge_pipeline: bool = True  # Enable knowledge enrichment
    store_findings_as_facts: bool = True  # Store findings in knowledge base
    knowledge_workspace_id: str = "default"  # Workspace for knowledge operations


class DocumentAuditor:
    """
    Orchestrates multi-agent document auditing.

    Uses large-context models for initial scanning and
    debate-style verification for findings.
    """

    def __init__(
        self,
        config: Optional[AuditConfig] = None,
        on_finding: Optional[Callable[[AuditFinding], None]] = None,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
    ):
        """
        Initialize document auditor.

        Args:
            config: Audit configuration
            on_finding: Callback when finding is detected
            on_progress: Callback for progress updates (session_id, progress, phase)
        """
        self.config = config or AuditConfig()
        self.on_finding = on_finding
        self.on_progress = on_progress

        # Session storage
        self._sessions: dict[str, AuditSession] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}

        # Audit type handlers
        self._handlers: dict[AuditType, Any] = {}
        self._load_handlers()

        # Knowledge pipeline integration
        self._knowledge_adapter: Optional[Any] = None
        if self.config.use_knowledge_pipeline:
            self._init_knowledge_adapter()

    def _load_handlers(self) -> None:
        """Load audit type handlers."""
        try:
            from aragora.audit.audit_types.security import SecurityAuditor

            self._handlers[AuditType.SECURITY] = SecurityAuditor()
        except ImportError:
            logger.debug("Security auditor not available")

        try:
            from aragora.audit.audit_types.compliance import ComplianceAuditor

            self._handlers[AuditType.COMPLIANCE] = ComplianceAuditor()
        except ImportError:
            logger.debug("Compliance auditor not available")

        try:
            from aragora.audit.audit_types.consistency import ConsistencyAuditor

            self._handlers[AuditType.CONSISTENCY] = ConsistencyAuditor()
        except ImportError:
            logger.debug("Consistency auditor not available")

        try:
            from aragora.audit.audit_types.quality import QualityAuditor

            self._handlers[AuditType.QUALITY] = QualityAuditor()
        except ImportError:
            logger.debug("Quality auditor not available")

    def _init_knowledge_adapter(self) -> None:
        """Initialize knowledge pipeline adapter."""
        try:
            from aragora.audit.knowledge_adapter import (
                AuditKnowledgeAdapter,
                KnowledgeAuditConfig,
            )

            kb_config = KnowledgeAuditConfig(
                enrich_with_facts=True,
                store_findings_as_facts=self.config.store_findings_as_facts,
                workspace_id=self.config.knowledge_workspace_id,
            )
            self._knowledge_adapter = AuditKnowledgeAdapter(kb_config)
            logger.info("Knowledge adapter initialized for audit")
        except ImportError:
            logger.debug("Knowledge adapter not available")

    async def create_session(
        self,
        document_ids: list[str],
        audit_types: Optional[list[str]] = None,
        name: str = "",
        description: str = "",
        model: Optional[str] = None,
        created_by: str = "",
        org_id: str = "",
    ) -> AuditSession:
        """
        Create a new audit session.

        Args:
            document_ids: Documents to audit
            audit_types: Types of audits to run
            name: Session name
            description: Session description
            model: Override primary model
            created_by: User creating session
            org_id: Organization ID

        Returns:
            Created audit session
        """
        # Parse audit types
        types = [AuditType.ALL]
        if audit_types:
            types = [AuditType(t) for t in audit_types]

        session = AuditSession(
            name=name or f"Audit {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            description=description,
            document_ids=document_ids,
            audit_types=types,
            model=model or self.config.primary_model,
            max_tokens_per_call=self.config.max_tokens_per_call,
            created_by=created_by,
            org_id=org_id,
        )

        self._sessions[session.id] = session
        logger.info(f"Created audit session {session.id} for {len(document_ids)} documents")

        return session

    def get_session(self, session_id: str) -> Optional[AuditSession]:
        """Get an audit session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        org_id: Optional[str] = None,
        status: Optional[AuditStatus] = None,
        limit: int = 100,
    ) -> list[AuditSession]:
        """List audit sessions with optional filtering."""
        sessions = list(self._sessions.values())

        if org_id:
            sessions = [s for s in sessions if s.org_id == org_id]
        if status:
            sessions = [s for s in sessions if s.status == status]

        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[:limit]

    async def run_audit(self, session_id: str) -> AuditSession:
        """
        Run the audit for a session.

        Args:
            session_id: Session to run

        Returns:
            Completed session with findings
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.status == AuditStatus.RUNNING:
            raise ValueError(f"Session {session_id} is already running")

        # Start audit
        session.status = AuditStatus.RUNNING
        session.started_at = datetime.now(timezone.utc)
        session.current_phase = "initializing"

        try:
            await self._execute_audit(session)
            session.status = AuditStatus.COMPLETED
        except asyncio.CancelledError:
            session.status = AuditStatus.CANCELLED
            logger.info(f"Audit session {session_id} cancelled")
        except Exception as e:
            session.status = AuditStatus.FAILED
            session.errors.append(str(e))
            logger.error(f"Audit session {session_id} failed: {e}")
            raise
        finally:
            session.completed_at = datetime.now(timezone.utc)

        return session

    async def _execute_audit(self, session: AuditSession) -> None:
        """Execute the audit pipeline.

        For multi-document audits (>1 document), uses Hive-Mind architecture
        for parallel processing with Queen-Worker orchestration.
        """
        # Phase 1: Load and prepare documents
        session.current_phase = "loading_documents"
        self._notify_progress(session, 0.05)

        chunks = await self._load_document_chunks(session)
        session.total_chunks = len(chunks)

        # Use Hive-Mind for multi-document audits when enabled
        use_hive_mind = (
            len(session.document_ids) > 1 and self.config.use_hive_mind and len(chunks) > 1
        )

        if use_hive_mind:
            # Use Hive-Mind parallel orchestration
            verified_findings = await self._execute_with_hive_mind(session, chunks)
        else:
            # Use standard sequential pipeline
            verified_findings = await self._execute_standard_pipeline(session, chunks)

        # Phase 5: Store findings in knowledge base
        if self._knowledge_adapter and self.config.store_findings_as_facts:
            session.current_phase = "storing_knowledge"
            self._notify_progress(session, 0.95)
            try:
                stored_count = await self._knowledge_adapter.store_session_findings(session)
                logger.info(f"Stored {stored_count} findings as facts in knowledge base")
            except Exception as e:
                logger.warning(f"Failed to store findings in knowledge base: {e}")
                session.errors.append(f"Knowledge storage error: {e}")

        # Phase 6: Finalize
        session.current_phase = "finalizing"
        session.findings = verified_findings
        session.progress = 1.0
        self._notify_progress(session, 1.0)

        logger.info(f"Audit session {session.id} completed: {len(verified_findings)} findings")

    async def _execute_standard_pipeline(
        self,
        session: AuditSession,
        chunks: list[dict[str, Any]],
    ) -> list[AuditFinding]:
        """Execute standard sequential audit pipeline."""
        # Phase 2: Initial scan with large context model
        session.current_phase = "initial_scan"
        self._notify_progress(session, 0.1)

        initial_findings = await self._initial_scan(session, chunks)

        # Phase 3: Detailed analysis per audit type
        session.current_phase = "detailed_analysis"
        self._notify_progress(session, 0.3)

        for audit_type in self._get_effective_audit_types(session):
            type_findings = await self._run_type_audit(session, chunks, audit_type)
            initial_findings.extend(type_findings)

        # Phase 4: Multi-agent verification
        session.current_phase = "verification"
        self._notify_progress(session, 0.7)

        return await self._verify_findings(session, initial_findings)

    async def _execute_with_hive_mind(
        self,
        session: AuditSession,
        chunks: list[dict[str, Any]],
    ) -> list[AuditFinding]:
        """Execute audit using Hive-Mind parallel orchestration.

        Uses Queen-Worker model for parallel document processing:
        - Queen decomposes audit into tasks
        - Workers process documents in parallel by specialty
        - Findings aggregated and verified via Byzantine consensus
        """
        try:
            from aragora.audit.hive_mind import (
                AuditHiveMind,
                HiveMindConfig,
            )

            session.current_phase = "hive_mind_init"
            self._notify_progress(session, 0.1)

            # Configure Hive-Mind
            hive_config = HiveMindConfig(
                max_concurrent_workers=min(8, len(chunks)),
                task_timeout_seconds=self.config.chunk_timeout_seconds,
                verify_critical_findings=self.config.consensus_verification,
            )

            # Create default agents for Hive-Mind orchestration
            from aragora.agents.api_agents.gemini import GeminiAgent

            queen_agent = GeminiAgent(
                name="audit_queen",
                model=session.model,
            )
            # Create worker agents - use multiple for parallelism
            worker_agents = [
                GeminiAgent(name=f"audit_worker_{i}", model=session.model)
                for i in range(min(4, len(chunks)))
            ]

            # Create Hive-Mind orchestrator
            hive_mind = AuditHiveMind(
                queen=queen_agent,
                workers=worker_agents,
                config=hive_config,
            )

            # Progress callback is configured via hive_config.progress_callback
            # Wire progress callback via config
            if self.on_progress:

                def progress_cb(phase: str, completed: int, total: int) -> None:
                    if total > 0:
                        progress = 0.1 + ((completed / total) * 0.8)
                        self._notify_progress(session, progress)

                hive_config.progress_callback = progress_cb

            # Execute with Hive-Mind
            session.current_phase = "hive_mind_execution"
            result = await hive_mind.audit_documents(session, chunks)

            return result.findings

        except ImportError:
            logger.warning("Hive-Mind not available, falling back to standard pipeline")
            return await self._execute_standard_pipeline(session, chunks)

        except Exception as e:
            logger.error(f"Hive-Mind execution failed: {e}, falling back to standard")
            session.errors.append(f"Hive-Mind error: {e}")
            return await self._execute_standard_pipeline(session, chunks)

    async def _load_document_chunks(
        self,
        session: AuditSession,
    ) -> list[dict[str, Any]]:
        """Load and chunk documents for analysis."""
        chunks = []

        try:
            from aragora.documents.chunking import get_context_manager

            _context_manager = get_context_manager()  # Verify module is available

            # For now, create placeholder chunks
            # In production, this would load from document store
            for doc_id in session.document_ids:
                # Placeholder - real implementation would fetch from storage
                chunks.append(
                    {
                        "id": f"{doc_id}_chunk_0",
                        "document_id": doc_id,
                        "content": f"Document {doc_id} content placeholder",
                        "sequence": 0,
                    }
                )

        except ImportError:
            logger.warning("Document chunking not available, using basic loading")
            for doc_id in session.document_ids:
                chunks.append(
                    {
                        "id": f"{doc_id}_chunk_0",
                        "document_id": doc_id,
                        "content": f"Document {doc_id} content placeholder",
                        "sequence": 0,
                    }
                )

        return chunks

    async def _initial_scan(
        self,
        session: AuditSession,
        chunks: list[dict[str, Any]],
    ) -> list[AuditFinding]:
        """
        Perform initial scan with large context model.

        Uses Gemini 3 Pro (1M tokens) for comprehensive scanning.
        """
        findings = []

        try:
            from aragora.agents.api_agents.gemini import GeminiAgent

            agent = GeminiAgent(
                name="initial_scanner",
                model=session.model,
            )

            # Build context from chunks
            all_content = "\n\n---\n\n".join(c.get("content", "") for c in chunks)

            # Scan prompt
            scan_prompt = f"""Analyze these documents for potential issues:

{all_content}

Look for:
1. Security vulnerabilities (exposed credentials, injection risks)
2. Compliance issues (PII handling, missing disclosures)
3. Inconsistencies (contradictions, outdated information)
4. Quality problems (ambiguity, missing information)

For each issue found, provide:
- Category (security/compliance/consistency/quality)
- Severity (critical/high/medium/low)
- Title (brief description)
- Evidence (quoted text)
- Location (which document/section)
- Recommendation

Format as JSON array of findings."""

            response = await agent.generate(scan_prompt)

            # Parse findings from response
            findings = self._parse_findings_from_response(
                response,
                session.id,
                "initial_scanner",
            )

        except Exception as e:
            logger.error(f"Initial scan failed: {e}")
            session.errors.append(f"Initial scan error: {e}")

        return findings

    async def _run_type_audit(
        self,
        session: AuditSession,
        chunks: list[dict[str, Any]],
        audit_type: AuditType,
    ) -> list[AuditFinding]:
        """Run specific type of audit."""
        handler = self._handlers.get(audit_type)
        if not handler:
            logger.debug(f"No handler for audit type {audit_type.value}")
            return []

        try:
            return await handler.audit(chunks, session)
        except Exception as e:
            logger.error(f"{audit_type.value} audit failed: {e}")
            session.errors.append(f"{audit_type.value} audit error: {e}")
            return []

    async def _verify_findings(
        self,
        session: AuditSession,
        findings: list[AuditFinding],
    ) -> list[AuditFinding]:
        """
        Verify findings using multi-agent verification.

        Uses debate-style verification where multiple agents
        examine and confirm/dispute findings.
        """
        if not self.config.require_confirmation:
            return findings

        verified = []

        try:
            from aragora.agents.api_agents.anthropic import AnthropicAPIAgent
            from aragora.agents.api_agents.openai import OpenAIAPIAgent

            verifier = AnthropicAPIAgent(
                name="verifier",
                model=self.config.verification_model,
            )
            adversary = OpenAIAPIAgent(
                name="adversary",
                model=self.config.adversarial_model,
            )

            for finding in findings:
                # Skip low-confidence findings
                if finding.confidence < self.config.min_confidence:
                    continue

                # Get verification
                verify_prompt = f"""Review this audit finding for accuracy:

Title: {finding.title}
Category: {finding.category}
Severity: {finding.severity.value}
Evidence: {finding.evidence_text}

Is this a valid finding? Respond with:
- CONFIRMED: [reason] if the finding is valid
- DISPUTED: [reason] if the finding is invalid or false positive
- NEEDS_INFO: [question] if more context is needed"""

                verify_response = await verifier.generate(verify_prompt)

                if "CONFIRMED" in verify_response.upper():
                    finding.confirmed_by.append("verifier")

                    # Also check with adversary
                    adversary_response = await adversary.generate(verify_prompt)
                    if "CONFIRMED" in adversary_response.upper():
                        finding.confirmed_by.append("adversary")
                    elif "DISPUTED" in adversary_response.upper():
                        finding.disputed_by.append("adversary")

                elif "DISPUTED" in verify_response.upper():
                    finding.disputed_by.append("verifier")

                # Include if enough confirmations
                confirmations = len(finding.confirmed_by)
                disputes = len(finding.disputed_by)

                if confirmations >= self.config.confirmation_threshold:
                    verified.append(finding)
                elif confirmations > disputes:
                    # Partial confirmation - include with lower confidence
                    finding.confidence *= 0.8
                    verified.append(finding)

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            # Return unverified findings if verification fails
            return findings

        return verified

    def _get_effective_audit_types(self, session: AuditSession) -> list[AuditType]:
        """Get the actual audit types to run."""
        if AuditType.ALL in session.audit_types:
            return [
                AuditType.SECURITY,
                AuditType.COMPLIANCE,
                AuditType.CONSISTENCY,
                AuditType.QUALITY,
            ]
        return session.audit_types

    def _parse_findings_from_response(
        self,
        response: str,
        session_id: str,
        agent_name: str,
    ) -> list[AuditFinding]:
        """Parse findings from agent response."""
        import json
        import re

        findings = []

        # Try to extract JSON array from response
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for item in data:
                    finding = AuditFinding(
                        session_id=session_id,
                        document_id=item.get("document_id", ""),
                        audit_type=AuditType(item.get("category", "quality")),
                        category=item.get("category", ""),
                        severity=FindingSeverity(item.get("severity", "medium").lower()),
                        confidence=item.get("confidence", 0.8),
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        evidence_text=item.get("evidence", ""),
                        evidence_location=item.get("location", ""),
                        recommendation=item.get("recommendation", ""),
                        found_by=agent_name,
                    )
                    findings.append(finding)

                    if self.on_finding:
                        self.on_finding(finding)

            except json.JSONDecodeError:
                logger.warning("Failed to parse findings JSON from response")

        return findings

    def _notify_progress(self, session: AuditSession, progress: float) -> None:
        """Notify progress callback."""
        session.progress = progress
        if self.on_progress:
            self.on_progress(session.id, progress, session.current_phase)

    async def pause_audit(self, session_id: str) -> bool:
        """Pause a running audit."""
        session = self._sessions.get(session_id)
        if not session or session.status != AuditStatus.RUNNING:
            return False

        task = self._running_tasks.get(session_id)
        if task:
            task.cancel()

        session.status = AuditStatus.PAUSED
        return True

    async def resume_audit(self, session_id: str) -> AuditSession:
        """Resume a paused audit."""
        session = self._sessions.get(session_id)
        if not session or session.status != AuditStatus.PAUSED:
            raise ValueError(f"Cannot resume session {session_id}")

        return await self.run_audit(session_id)

    async def cancel_audit(self, session_id: str) -> bool:
        """Cancel an audit session."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        if session.status == AuditStatus.RUNNING:
            task = self._running_tasks.get(session_id)
            if task:
                task.cancel()

        session.status = AuditStatus.CANCELLED
        session.completed_at = datetime.now(timezone.utc)
        return True

    def get_findings(
        self,
        session_id: str,
        severity: Optional[FindingSeverity] = None,
        audit_type: Optional[AuditType] = None,
        status: Optional[FindingStatus] = None,
    ) -> list[AuditFinding]:
        """Get findings for a session with optional filtering."""
        session = self._sessions.get(session_id)
        if not session:
            return []

        findings = session.findings

        if severity:
            findings = [f for f in findings if f.severity == severity]
        if audit_type:
            findings = [f for f in findings if f.audit_type == audit_type]
        if status:
            findings = [f for f in findings if f.status == status]

        return findings

    def update_finding_status(
        self,
        session_id: str,
        finding_id: str,
        status: FindingStatus,
        note: str = "",
    ) -> bool:
        """Update the status of a finding."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        for finding in session.findings:
            if finding.id == finding_id:
                finding.status = status
                finding.updated_at = datetime.now(timezone.utc)
                if note:
                    finding.tags.append(f"status_note:{note}")
                return True

        return False


# Global instance
_auditor: Optional[DocumentAuditor] = None


def get_document_auditor(config: Optional[AuditConfig] = None) -> DocumentAuditor:
    """Get or create global document auditor instance."""
    global _auditor
    if _auditor is None:
        _auditor = DocumentAuditor(config)
    return _auditor


__all__ = [
    "DocumentAuditor",
    "AuditSession",
    "AuditConfig",
    "AuditFinding",
    "AuditType",
    "AuditStatus",
    "FindingSeverity",
    "FindingStatus",
    "get_document_auditor",
]
