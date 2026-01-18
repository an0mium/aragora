"""
Legal document analysis for law firms.

Detects:
- Contract clause risks (indemnification, liability, termination)
- Obligation tracking (deadlines, deliverables, conditions)
- Conflict of interest indicators
- Jurisdiction and governing law issues
- Missing standard clauses
- Ambiguous language requiring clarification
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence, TypedDict

from ..base_auditor import AuditorCapabilities, AuditContext, BaseAuditor, ChunkData
from ..document_auditor import (
    AuditFinding,
    AuditType,
    FindingSeverity,
)


class RequiredClauseDict(TypedDict):
    """Type definition for required clause entries."""

    name: str
    keywords: list[str]
    severity: FindingSeverity
    description: str
    recommendation: str

logger = logging.getLogger(__name__)


@dataclass
class LegalPattern:
    """A pattern for detecting legal issues."""

    name: str
    pattern: re.Pattern
    severity: FindingSeverity
    category: str
    description: str
    recommendation: str
    clause_type: str = ""  # For categorization


@dataclass
class ObligationPattern:
    """Pattern for extracting obligations from contracts."""

    name: str
    trigger_words: list[str]
    obligation_type: str  # "shall", "must", "will", "may"
    party_indicators: list[str]
    deadline_patterns: list[str]


@dataclass
class ExtractedObligation:
    """An obligation extracted from a contract."""

    text: str
    obligation_type: str
    obligated_party: Optional[str]
    deadline: Optional[str]
    condition: Optional[str]
    location: str


class LegalAuditor(BaseAuditor):
    """
    Audits legal documents for risks, obligations, and compliance issues.

    Specialized for:
    - Contract review and due diligence
    - Obligation extraction and tracking
    - Risk identification in legal agreements
    - Compliance with standard legal practices
    """

    # High-risk clause patterns
    RISK_PATTERNS = [
        LegalPattern(
            name="Unlimited Liability",
            pattern=re.compile(
                r"(?i)(unlimited\s+liability|no\s+cap\s+on\s+(liability|damages)|"
                r"liable\s+for\s+all\s+(damages|losses)|without\s+limitation)"
            ),
            severity=FindingSeverity.CRITICAL,
            category="liability_risk",
            description="Unlimited liability clause detected - significant financial exposure",
            recommendation="Negotiate a liability cap (typically 12-24 months of fees or contract value)",
            clause_type="liability",
        ),
        LegalPattern(
            name="Broad Indemnification",
            pattern=re.compile(
                r"(?i)(indemnify|hold\s+harmless|defend).{0,100}"
                r"(any\s+and\s+all|all\s+claims|all\s+losses|any\s+claims|"
                r"third.party|arising\s+from\s+any)"
            ),
            severity=FindingSeverity.HIGH,
            category="indemnification_risk",
            description="Broad indemnification clause may create unlimited exposure",
            recommendation="Limit indemnification to direct damages and exclude consequential damages",
            clause_type="indemnification",
        ),
        LegalPattern(
            name="Unilateral Termination",
            pattern=re.compile(
                r"(?i)(may\s+terminate.{0,50}(at\s+any\s+time|for\s+any\s+reason|"
                r"without\s+cause|in\s+its\s+sole\s+discretion)|"
                r"termination\s+for\s+convenience)"
            ),
            severity=FindingSeverity.MEDIUM,
            category="termination_risk",
            description="One-sided termination rights without reciprocity",
            recommendation="Ensure termination rights are mutual or add cure period",
            clause_type="termination",
        ),
        LegalPattern(
            name="Auto-Renewal Trap",
            pattern=re.compile(
                r"(?i)(auto.?renew|automatic.?renewal|renews?\s+automatically).{0,100}"
                r"(unless.{0,50}(notice|written|days?\s+prior)|"
                r"for\s+(successive|additional|renewal)\s+(term|period))"
            ),
            severity=FindingSeverity.MEDIUM,
            category="renewal_risk",
            description="Auto-renewal clause may lock party into extended commitments",
            recommendation="Calendar the notice deadline; negotiate shorter notice period",
            clause_type="term",
        ),
        LegalPattern(
            name="Assignment Without Consent",
            pattern=re.compile(
                r"(?i)(may\s+assign|freely\s+assign|assign.{0,30}without).{0,50}"
                r"(consent|approval|notice)"
            ),
            severity=FindingSeverity.MEDIUM,
            category="assignment_risk",
            description="Assignment rights may transfer obligations to unknown party",
            recommendation="Require written consent for assignment",
            clause_type="assignment",
        ),
        LegalPattern(
            name="Consequential Damages Exposure",
            pattern=re.compile(
                r"(?i)(consequential|indirect|special|punitive|incidental)\s+damages"
            ),
            severity=FindingSeverity.HIGH,
            category="damages_risk",
            description="Exposure to consequential/indirect damages",
            recommendation="Add mutual exclusion of consequential damages",
            clause_type="damages",
        ),
        LegalPattern(
            name="Non-Compete Concern",
            pattern=re.compile(
                r"(?i)(non.?compete|not\s+compete|refrain\s+from\s+competing|"
                r"competitive\s+activity).{0,100}(years?|months?|period)"
            ),
            severity=FindingSeverity.HIGH,
            category="restrictive_covenant",
            description="Non-compete clause may restrict future business activities",
            recommendation="Review scope, duration, and geographic limitations",
            clause_type="non_compete",
        ),
        LegalPattern(
            name="Non-Solicitation Clause",
            pattern=re.compile(
                r"(?i)(non.?solicit|not\s+solicit|refrain\s+from\s+soliciting).{0,100}"
                r"(employee|customer|client|personnel)"
            ),
            severity=FindingSeverity.MEDIUM,
            category="restrictive_covenant",
            description="Non-solicitation clause restricts hiring/business development",
            recommendation="Negotiate reasonable duration (typically 12-24 months)",
            clause_type="non_solicit",
        ),
        LegalPattern(
            name="Exclusivity Requirement",
            pattern=re.compile(
                r"(?i)(exclusive|exclusivity|sole\s+(provider|supplier|vendor)|"
                r"not\s+engage.{0,30}(other|competitor|third))"
            ),
            severity=FindingSeverity.HIGH,
            category="exclusivity_risk",
            description="Exclusivity clause limits ability to work with others",
            recommendation="Define scope narrowly; add performance requirements",
            clause_type="exclusivity",
        ),
        LegalPattern(
            name="IP Assignment Concern",
            pattern=re.compile(
                r"(?i)(assign|transfer|convey).{0,50}"
                r"(intellectual\s+property|IP|patent|copyright|trademark|"
                r"work\s+product|inventions?|developments?)"
            ),
            severity=FindingSeverity.HIGH,
            category="ip_risk",
            description="IP assignment clause may transfer valuable rights",
            recommendation="Ensure assignment is limited to deliverables; retain background IP",
            clause_type="ip_assignment",
        ),
        LegalPattern(
            name="Audit Rights",
            pattern=re.compile(
                r"(?i)(audit|inspect|examine).{0,50}" r"(books|records|facilities|premises|systems)"
            ),
            severity=FindingSeverity.LOW,
            category="audit_rights",
            description="Audit rights clause - review scope and notice requirements",
            recommendation="Ensure reasonable notice period and scope limitations",
            clause_type="audit",
        ),
    ]

    # Missing clause detection
    REQUIRED_CLAUSES: list[RequiredClauseDict] = [
        {
            "name": "Limitation of Liability",
            "keywords": ["limitation of liability", "liability cap", "aggregate liability"],
            "severity": FindingSeverity.HIGH,
            "description": "Contract lacks limitation of liability clause",
            "recommendation": "Add mutual limitation of liability (typically contract value or 12 months fees)",
        },
        {
            "name": "Confidentiality",
            "keywords": ["confidential", "confidentiality", "non-disclosure", "proprietary"],
            "severity": FindingSeverity.MEDIUM,
            "description": "Contract lacks confidentiality provisions",
            "recommendation": "Add mutual confidentiality clause with standard carve-outs",
        },
        {
            "name": "Governing Law",
            "keywords": ["governing law", "governed by", "laws of", "jurisdiction"],
            "severity": FindingSeverity.MEDIUM,
            "description": "Contract does not specify governing law",
            "recommendation": "Specify governing law favorable to your jurisdiction",
        },
        {
            "name": "Dispute Resolution",
            "keywords": ["dispute", "arbitration", "mediation", "litigation", "court"],
            "severity": FindingSeverity.MEDIUM,
            "description": "No dispute resolution mechanism specified",
            "recommendation": "Add dispute resolution clause (negotiation > mediation > arbitration/litigation)",
        },
        {
            "name": "Force Majeure",
            "keywords": ["force majeure", "act of god", "beyond control", "unforeseeable"],
            "severity": FindingSeverity.LOW,
            "description": "Contract lacks force majeure clause",
            "recommendation": "Add force majeure clause covering pandemic, natural disasters, etc.",
        },
        {
            "name": "Notice Provisions",
            "keywords": ["notice", "written notice", "notify", "notification"],
            "severity": FindingSeverity.LOW,
            "description": "Contract lacks formal notice provisions",
            "recommendation": "Add notice clause specifying method and addresses",
        },
    ]

    # Ambiguous language patterns
    AMBIGUITY_PATTERNS = [
        LegalPattern(
            name="Vague Reasonableness",
            pattern=re.compile(r"(?i)(reasonable|reasonably)\s+(efforts?|time|manner|notice)"),
            severity=FindingSeverity.LOW,
            category="ambiguity",
            description="'Reasonable' standard may lead to interpretation disputes",
            recommendation="Define specific metrics or timeframes where possible",
            clause_type="standard",
        ),
        LegalPattern(
            name="Best Efforts Ambiguity",
            pattern=re.compile(r"(?i)best\s+efforts?|commercially\s+reasonable"),
            severity=FindingSeverity.LOW,
            category="ambiguity",
            description="'Best efforts' or 'commercially reasonable' standard is subjective",
            recommendation="Define specific obligations or success criteria",
            clause_type="standard",
        ),
        LegalPattern(
            name="Material Undefined",
            pattern=re.compile(r"(?i)material\s+(breach|change|adverse|impact)"),
            severity=FindingSeverity.LOW,
            category="ambiguity",
            description="'Material' term not defined - may lead to disputes",
            recommendation="Define materiality threshold (e.g., percentage, dollar amount)",
            clause_type="definition",
        ),
        LegalPattern(
            name="Promptly Undefined",
            pattern=re.compile(
                r"(?i)(promptly|immediately|without\s+delay)\s+(notify|inform|provide)"
            ),
            severity=FindingSeverity.LOW,
            category="ambiguity",
            description="Timing terms are ambiguous without specific timeframes",
            recommendation="Replace with specific timeframes (e.g., 'within 5 business days')",
            clause_type="timing",
        ),
    ]

    # Obligation extraction patterns
    OBLIGATION_INDICATORS = [
        ObligationPattern(
            name="Shall Obligation",
            trigger_words=["shall", "must", "will", "agrees to", "covenants to"],
            obligation_type="mandatory",
            party_indicators=["party", "company", "vendor", "customer", "licensor", "licensee"],
            deadline_patterns=[
                r"within\s+(\d+)\s+(days?|business\s+days?|weeks?|months?)",
                r"by\s+(\w+\s+\d+,?\s+\d{4})",
                r"no\s+later\s+than",
                r"prior\s+to",
            ],
        ),
        ObligationPattern(
            name="May Permission",
            trigger_words=["may", "is entitled to", "has the right to", "is permitted to"],
            obligation_type="permissive",
            party_indicators=["party", "company", "vendor", "customer"],
            deadline_patterns=[],
        ),
    ]

    def __init__(self) -> None:
        """Initialize the legal auditor."""
        self.obligations: list[ExtractedObligation] = []

    @property
    def audit_type_id(self) -> str:
        return "legal"

    @property
    def display_name(self) -> str:
        return "Legal Contract Analysis"

    @property
    def description(self) -> str:
        return (
            "Analyzes legal contracts for risks, obligations, and compliance issues. "
            "Detects liability risks, indemnification clauses, termination terms, "
            "missing standard clauses, and ambiguous language."
        )

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> AuditorCapabilities:
        return AuditorCapabilities(
            supports_chunk_analysis=True,
            supports_cross_document=True,
            supports_streaming=False,
            requires_llm=True,
            supported_doc_types=["pdf", "docx", "doc", "txt", "md", "rtf"],
            max_chunk_size=8000,
            finding_categories=[
                "liability_risk",
                "indemnification_risk",
                "termination_risk",
                "renewal_risk",
                "assignment_risk",
                "damages_risk",
                "restrictive_covenant",
                "exclusivity_risk",
                "ip_risk",
                "audit_rights",
                "missing_clause",
                "ambiguity",
            ],
            custom_capabilities={
                "obligation_extraction": True,
                "clause_detection": True,
                "missing_clause_analysis": True,
                "ambiguity_detection": True,
            },
        )

    async def analyze_chunk(
        self,
        chunk: ChunkData,
        context: AuditContext,
    ) -> list[AuditFinding]:
        """
        Analyze a document chunk for legal issues.

        Args:
            chunk: The chunk to analyze
            context: Audit context with session info and utilities

        Returns:
            List of findings from this chunk
        """
        findings: list[AuditFinding] = []
        content = chunk.content

        # Check risk patterns
        for pattern in self.RISK_PATTERNS:
            matches = pattern.pattern.finditer(content)
            for match in matches:
                findings.append(
                    AuditFinding(
                        session_id=context.session.id,
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        audit_type=AuditType.COMPLIANCE,
                        category=pattern.category,
                        severity=pattern.severity,
                        confidence=0.85,
                        title=pattern.name,
                        description=pattern.description,
                        evidence_text=self._get_context(content, match.start(), match.end()),
                        evidence_location=f"Chunk {chunk.id}, chars {match.start()}-{match.end()}",
                        recommendation=pattern.recommendation,
                        affected_scope="clause",
                        found_by="legal_auditor",
                        tags=[pattern.clause_type, "contract_risk"],
                    )
                )

        # Check ambiguity patterns
        for pattern in self.AMBIGUITY_PATTERNS:
            matches = pattern.pattern.finditer(content)
            for match in matches:
                findings.append(
                    AuditFinding(
                        session_id=context.session.id,
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        audit_type=AuditType.QUALITY,
                        category=pattern.category,
                        severity=pattern.severity,
                        confidence=0.7,
                        title=pattern.name,
                        description=pattern.description,
                        evidence_text=self._get_context(content, match.start(), match.end()),
                        evidence_location=f"Chunk {chunk.id}, chars {match.start()}-{match.end()}",
                        recommendation=pattern.recommendation,
                        affected_scope="clause",
                        found_by="legal_auditor",
                        tags=["ambiguity", "drafting"],
                    )
                )

        # Extract obligations
        self._extract_obligations(content, chunk.document_id, chunk.id)

        return findings

    async def cross_document_analysis(
        self,
        chunks: Sequence[ChunkData],
        context: AuditContext,
    ) -> list[AuditFinding]:
        """
        Analyze across documents for missing clauses and cross-references.

        Args:
            chunks: All chunks in the audit scope
            context: Audit context

        Returns:
            List of cross-document findings
        """
        findings: list[AuditFinding] = []

        # Combine all content for missing clause analysis
        all_content = " ".join(chunk.content for chunk in chunks)
        content_lower = all_content.lower()

        # Get the first document_id for findings (or use a placeholder)
        primary_doc_id = chunks[0].document_id if chunks else "unknown"

        # Check for missing standard clauses
        for clause in self.REQUIRED_CLAUSES:
            found = any(kw in content_lower for kw in clause["keywords"])
            if not found:
                findings.append(
                    AuditFinding(
                        session_id=context.session.id,
                        document_id=primary_doc_id,
                        chunk_id=None,
                        audit_type=AuditType.COMPLIANCE,
                        category="missing_clause",
                        severity=clause["severity"],
                        confidence=0.9,
                        title=f"Missing {clause['name']} Clause",
                        description=clause["description"],
                        evidence_text="",
                        evidence_location="document-wide analysis",
                        recommendation=clause["recommendation"],
                        affected_scope="document",
                        found_by="legal_auditor",
                        tags=["missing_clause", "contract_review"],
                    )
                )

        # Add obligation summary as informational finding if obligations were found
        if self.obligations:
            mandatory = [o for o in self.obligations if o.obligation_type == "mandatory"]
            with_deadlines = [o for o in mandatory if o.deadline]

            if mandatory:
                findings.append(
                    AuditFinding(
                        session_id=context.session.id,
                        document_id=primary_doc_id,
                        chunk_id=None,
                        audit_type=AuditType.QUALITY,
                        category="obligation_summary",
                        severity=FindingSeverity.INFO,
                        confidence=0.95,
                        title="Contract Obligations Summary",
                        description=(
                            f"Found {len(mandatory)} mandatory obligations, "
                            f"{len(with_deadlines)} with explicit deadlines. "
                            "Review to ensure all obligations are tracked."
                        ),
                        evidence_text="\n".join(f"- {o.text[:100]}..." for o in mandatory[:5]),
                        evidence_location="cross-document analysis",
                        recommendation="Create obligation tracking for all mandatory requirements",
                        affected_scope="document",
                        found_by="legal_auditor",
                        tags=["obligations", "tracking"],
                    )
                )

        return findings

    def _extract_obligations(
        self,
        content: str,
        document_id: str,
        chunk_id: str,
    ) -> None:
        """Extract obligations from contract text."""
        for indicator in self.OBLIGATION_INDICATORS:
            for trigger in indicator.trigger_words:
                pattern = re.compile(
                    rf"(?i)([A-Z][^.]*\b{trigger}\b[^.]+\.)",
                    re.MULTILINE,
                )
                for match in pattern.finditer(content):
                    sentence = match.group(1)

                    # Try to identify the obligated party
                    party = None
                    for party_word in indicator.party_indicators:
                        if party_word.lower() in sentence.lower():
                            party = party_word
                            break

                    # Try to extract deadline
                    deadline = None
                    for deadline_pattern in indicator.deadline_patterns:
                        deadline_match = re.search(deadline_pattern, sentence, re.IGNORECASE)
                        if deadline_match:
                            deadline = deadline_match.group(0)
                            break

                    self.obligations.append(
                        ExtractedObligation(
                            text=sentence.strip(),
                            obligation_type=indicator.obligation_type,
                            obligated_party=party,
                            deadline=deadline,
                            condition=None,
                            location=f"{document_id}:{chunk_id}",
                        )
                    )

    def get_obligation_summary(self) -> dict[str, Any]:
        """Get a summary of extracted obligations."""
        mandatory = [o for o in self.obligations if o.obligation_type == "mandatory"]
        with_deadlines = [o for o in mandatory if o.deadline]

        return {
            "total_obligations": len(self.obligations),
            "mandatory_obligations": len(mandatory),
            "obligations_with_deadlines": len(with_deadlines),
            "obligations": [
                {
                    "text": o.text[:200],
                    "type": o.obligation_type,
                    "party": o.obligated_party,
                    "deadline": o.deadline,
                    "location": o.location,
                }
                for o in self.obligations[:50]  # Limit to first 50
            ],
        }

    def _get_context(self, content: str, start: int, end: int, context_chars: int = 100) -> str:
        """Get surrounding context for a match."""
        ctx_start = max(0, start - context_chars)
        ctx_end = min(len(content), end + context_chars)
        prefix = "..." if ctx_start > 0 else ""
        suffix = "..." if ctx_end < len(content) else ""
        return f"{prefix}{content[ctx_start:ctx_end]}{suffix}"

    async def pre_audit_hook(self, context: AuditContext) -> None:
        """Reset state before audit."""
        self.obligations = []


# Register with the audit registry on import
def register_legal_auditor() -> None:
    """Register the legal auditor with the global registry."""
    try:
        from ..registry import audit_registry

        audit_registry.register(LegalAuditor())
    except ImportError:
        pass  # Registry not available


__all__ = [
    "LegalAuditor",
    "LegalPattern",
    "ObligationPattern",
    "ExtractedObligation",
    "register_legal_auditor",
]
