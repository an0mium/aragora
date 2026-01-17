"""
Base Auditor Abstract Class.

Provides the foundation for pluggable audit types. Custom auditors
can be created by subclassing BaseAuditor and implementing the
required methods.

Usage:
    from aragora.audit.base_auditor import BaseAuditor, AuditContext

    class CustomAuditor(BaseAuditor):
        @property
        def audit_type_id(self) -> str:
            return "custom_legal"

        @property
        def display_name(self) -> str:
            return "Legal Contract Analysis"

        @property
        def description(self) -> str:
            return "Analyzes legal contracts for risks and obligations"

        async def analyze_chunk(self, chunk, context):
            # Implement chunk analysis
            return findings

        async def cross_document_analysis(self, chunks, context):
            # Implement cross-document analysis
            return findings
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from aragora.audit.document_auditor import (
    AuditFinding,
    AuditSession,
    AuditType,
    FindingSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class AuditContext:
    """
    Context provided to auditors during analysis.

    Contains session information, configuration, and utilities
    for creating findings.
    """

    session: AuditSession
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None

    # Configuration
    model: str = "claude-3.5-sonnet"
    max_findings_per_chunk: int = 50
    confidence_threshold: float = 0.5
    include_low_severity: bool = True

    # Custom parameters from preset/template
    custom_params: dict[str, Any] = field(default_factory=dict)

    # Runtime state
    total_chunks_processed: int = 0
    total_findings: int = 0

    def create_finding(
        self,
        *,
        document_id: str,
        title: str,
        description: str,
        severity: FindingSeverity,
        category: str,
        chunk_id: Optional[str] = None,
        confidence: float = 0.8,
        evidence_text: str = "",
        evidence_location: str = "",
        recommendation: str = "",
        affected_scope: str = "chunk",
        found_by: str = "",
        tags: Optional[list[str]] = None,
    ) -> AuditFinding:
        """
        Create an audit finding with proper context.

        This is the preferred way to create findings as it
        automatically sets session context.
        """
        return AuditFinding(
            session_id=self.session.id,
            document_id=document_id,
            chunk_id=chunk_id,
            audit_type=AuditType.QUALITY,  # Will be overridden by registry
            category=category,
            severity=severity,
            confidence=confidence,
            title=title,
            description=description,
            evidence_text=evidence_text,
            evidence_location=evidence_location,
            recommendation=recommendation,
            affected_scope=affected_scope,
            found_by=found_by or self.session.model,
            tags=tags or [],
        )


@dataclass
class ChunkData:
    """
    Normalized chunk data for auditor processing.

    Provides a clean interface for accessing chunk content
    regardless of the underlying storage format.
    """

    id: str
    document_id: str
    content: str
    chunk_type: str = "text"  # text, heading, table, code, list, image, formula
    page_number: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    heading_context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkData":
        """Create from dictionary (e.g., from database/API)."""
        return cls(
            id=data.get("id", ""),
            document_id=data.get("document_id", ""),
            content=data.get("content", ""),
            chunk_type=data.get("chunk_type", "text"),
            page_number=data.get("page_number"),
            char_start=data.get("char_start"),
            char_end=data.get("char_end"),
            heading_context=data.get("heading_context", []),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "heading_context": self.heading_context,
            "metadata": self.metadata,
        }


@dataclass
class AuditorCapabilities:
    """
    Declares what an auditor can do.

    Used by the registry to understand auditor requirements
    and capabilities.
    """

    # Analysis modes
    supports_chunk_analysis: bool = True
    supports_cross_document: bool = False
    supports_streaming: bool = False

    # Requirements
    requires_llm: bool = False
    requires_vector_search: bool = False
    min_chunk_size: int = 10
    max_chunk_size: int = 100000

    # Categories this auditor produces
    finding_categories: list[str] = field(default_factory=list)

    # Supported document types (empty = all)
    supported_document_types: list[str] = field(default_factory=list)


class BaseAuditor(ABC):
    """
    Abstract base class for all audit types.

    Subclass this to create custom audit types that can be
    registered with the audit system.

    Example:
        class ContractRiskAuditor(BaseAuditor):
            @property
            def audit_type_id(self) -> str:
                return "contract_risk"

            @property
            def display_name(self) -> str:
                return "Contract Risk Analysis"

            @property
            def description(self) -> str:
                return "Identifies risky clauses and obligations in contracts"

            @property
            def capabilities(self) -> AuditorCapabilities:
                return AuditorCapabilities(
                    supports_cross_document=True,
                    requires_llm=True,
                    finding_categories=["liability", "indemnification", "termination"],
                )

            async def analyze_chunk(self, chunk, context):
                # Implementation
                return []

            async def cross_document_analysis(self, chunks, context):
                # Cross-reference analysis
                return []
    """

    @property
    @abstractmethod
    def audit_type_id(self) -> str:
        """
        Unique identifier for this audit type.

        Should be lowercase with underscores (e.g., "security", "contract_risk").
        This is used in API calls and configuration.
        """
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """
        Human-readable name for the audit type.

        Shown in UI and reports (e.g., "Security Analysis", "Contract Risk").
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Description of what this auditor does.

        Shown in UI to help users understand when to use it.
        """
        ...

    @property
    def capabilities(self) -> AuditorCapabilities:
        """
        Declare the capabilities of this auditor.

        Override to specify requirements and features.
        """
        return AuditorCapabilities()

    @property
    def version(self) -> str:
        """
        Version of this auditor implementation.

        Useful for tracking which version produced findings.
        """
        return "1.0.0"

    @property
    def author(self) -> str:
        """Author or source of this auditor."""
        return "aragora"

    @abstractmethod
    async def analyze_chunk(
        self,
        chunk: ChunkData,
        context: AuditContext,
    ) -> list[AuditFinding]:
        """
        Analyze a single document chunk.

        This is the primary analysis method. It's called for each
        chunk in the document set.

        Args:
            chunk: The chunk to analyze
            context: Audit context with session info and utilities

        Returns:
            List of findings from this chunk
        """
        ...

    async def cross_document_analysis(
        self,
        chunks: Sequence[ChunkData],
        context: AuditContext,
    ) -> list[AuditFinding]:
        """
        Perform analysis across multiple chunks/documents.

        Override this for auditors that need to compare or
        correlate information across documents (e.g., consistency
        checking, cross-reference validation).

        Args:
            chunks: All chunks in the audit scope
            context: Audit context

        Returns:
            List of cross-document findings
        """
        return []

    async def pre_audit_hook(self, context: AuditContext) -> None:
        """
        Called before the audit begins.

        Override to perform initialization, load resources, etc.
        """
        pass

    async def post_audit_hook(
        self,
        findings: list[AuditFinding],
        context: AuditContext,
    ) -> list[AuditFinding]:
        """
        Called after all analysis is complete.

        Override to perform post-processing, deduplication,
        severity adjustment, etc.

        Args:
            findings: All findings from this auditor
            context: Audit context

        Returns:
            Processed findings (can filter, modify, or add)
        """
        return findings

    def validate_finding(self, finding: AuditFinding, context: AuditContext) -> bool:
        """
        Validate a finding before it's added to results.

        Override to add custom validation rules.

        Args:
            finding: The finding to validate
            context: Audit context

        Returns:
            True if finding should be included, False to filter out
        """
        # Filter by confidence threshold
        if finding.confidence < context.confidence_threshold:
            return False

        # Filter low severity if configured
        if not context.include_low_severity and finding.severity in (
            FindingSeverity.LOW,
            FindingSeverity.INFO,
        ):
            return False

        return True

    # Compatibility method for existing auditor interface
    async def audit(
        self,
        chunks: list[dict[str, Any]],
        session: AuditSession,
    ) -> list[AuditFinding]:
        """
        Legacy compatibility method.

        New auditors should implement analyze_chunk() instead.
        This wraps the new interface for backward compatibility.
        """
        context = AuditContext(session=session, model=session.model)
        findings = []

        await self.pre_audit_hook(context)

        # Analyze each chunk
        chunk_data_list = []
        for chunk_dict in chunks:
            chunk = ChunkData.from_dict(chunk_dict)
            chunk_data_list.append(chunk)

            chunk_findings = await self.analyze_chunk(chunk, context)
            for finding in chunk_findings:
                if self.validate_finding(finding, context):
                    # Set the audit type from this auditor
                    finding.audit_type = self._get_audit_type_enum()
                    findings.append(finding)

            context.total_chunks_processed += 1

        # Cross-document analysis if supported
        if self.capabilities.supports_cross_document:
            cross_findings = await self.cross_document_analysis(chunk_data_list, context)
            for finding in cross_findings:
                if self.validate_finding(finding, context):
                    finding.audit_type = self._get_audit_type_enum()
                    findings.append(finding)

        # Post-processing
        findings = await self.post_audit_hook(findings, context)

        context.total_findings = len(findings)
        return findings

    def _get_audit_type_enum(self) -> AuditType:
        """Get the AuditType enum for this auditor."""
        # Map to existing enum or use QUALITY as fallback for custom types
        type_map = {
            "security": AuditType.SECURITY,
            "compliance": AuditType.COMPLIANCE,
            "consistency": AuditType.CONSISTENCY,
            "quality": AuditType.QUALITY,
        }
        return type_map.get(self.audit_type_id, AuditType.QUALITY)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.audit_type_id}, version={self.version})>"
