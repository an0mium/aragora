"""
Regulatory compliance auditing for SOX, GDPR, PCI-DSS, and industry standards.

Detects:
- SOX compliance issues (financial controls, audit trails)
- GDPR violations (data subject rights, consent, data protection)
- PCI-DSS issues (cardholder data, security controls)
- Industry-specific regulatory requirements
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence

from aragora.audit.base_auditor import (
    AuditContext,
    AuditorCapabilities,
    BaseAuditor,
    ChunkData,
)
from aragora.audit.document_auditor import (
    AuditFinding,
    FindingSeverity,
)

logger = logging.getLogger(__name__)


class RegulatoryFramework(str, Enum):
    """Supported regulatory frameworks."""

    SOX = "sox"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    CCPA = "ccpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST = "nist"
    FEDRAMP = "fedramp"


@dataclass
class RegulatoryRequirement:
    """A regulatory requirement to check."""

    framework: RegulatoryFramework
    control_id: str
    title: str
    description: str
    patterns: List[re.Pattern[str]]
    violation_indicators: List[str]
    severity: FindingSeverity
    remediation: str


# SOX Requirements (Sarbanes-Oxley)
SOX_REQUIREMENTS = [
    RegulatoryRequirement(
        framework=RegulatoryFramework.SOX,
        control_id="SOX-302",
        title="CEO/CFO Certification",
        description="Financial statements must be certified by CEO and CFO",
        patterns=[
            re.compile(r"(?i)(financial\s+statement|quarterly\s+report|annual\s+report)"),
        ],
        violation_indicators=[
            "missing certification",
            "unsigned",
            "not certified",
            "pending approval",
        ],
        severity=FindingSeverity.CRITICAL,
        remediation="Ensure CEO/CFO certification is obtained before filing",
    ),
    RegulatoryRequirement(
        framework=RegulatoryFramework.SOX,
        control_id="SOX-404",
        title="Internal Controls Assessment",
        description="Internal controls over financial reporting must be documented and tested",
        patterns=[
            re.compile(r"(?i)(internal\s+control|control\s+environment|control\s+activity)"),
        ],
        violation_indicators=[
            "control weakness",
            "material weakness",
            "significant deficiency",
            "control failure",
            "override",
        ],
        severity=FindingSeverity.HIGH,
        remediation="Document control deficiency and implement remediation plan",
    ),
    RegulatoryRequirement(
        framework=RegulatoryFramework.SOX,
        control_id="SOX-802",
        title="Document Retention",
        description="Audit workpapers must be retained for 7 years",
        patterns=[
            re.compile(r"(?i)(audit\s+workpaper|retention|archive|destroy|delete)"),
        ],
        violation_indicators=[
            "delete",
            "destroy",
            "shred",
            "dispose",
            "remove",
        ],
        severity=FindingSeverity.HIGH,
        remediation="Implement proper document retention policies per SOX Section 802",
    ),
    RegulatoryRequirement(
        framework=RegulatoryFramework.SOX,
        control_id="SOX-806",
        title="Whistleblower Protection",
        description="Retaliation against whistleblowers is prohibited",
        patterns=[
            re.compile(r"(?i)(whistleblower|complaint|hotline|ethics\s+report)"),
        ],
        violation_indicators=[
            "retaliation",
            "terminate",
            "demote",
            "discipline",
            "threaten",
        ],
        severity=FindingSeverity.CRITICAL,
        remediation="Ensure whistleblower protection policies are in place and followed",
    ),
]

# GDPR Requirements
GDPR_REQUIREMENTS = [
    RegulatoryRequirement(
        framework=RegulatoryFramework.GDPR,
        control_id="GDPR-6",
        title="Lawful Basis for Processing",
        description="Personal data processing must have a lawful basis",
        patterns=[
            re.compile(r"(?i)(personal\s+data|pii|data\s+subject)"),
        ],
        violation_indicators=[
            "without consent",
            "no legal basis",
            "unauthorized processing",
        ],
        severity=FindingSeverity.HIGH,
        remediation="Document lawful basis for all personal data processing activities",
    ),
    RegulatoryRequirement(
        framework=RegulatoryFramework.GDPR,
        control_id="GDPR-7",
        title="Consent Requirements",
        description="Consent must be freely given, specific, informed, and unambiguous",
        patterns=[
            re.compile(r"(?i)(consent|opt[\s-]?in|permission|agree)"),
        ],
        violation_indicators=[
            "pre-checked",
            "bundled consent",
            "implicit",
            "assumed",
            "default",
        ],
        severity=FindingSeverity.HIGH,
        remediation="Implement granular consent mechanisms with clear affirmative action",
    ),
    RegulatoryRequirement(
        framework=RegulatoryFramework.GDPR,
        control_id="GDPR-15-22",
        title="Data Subject Rights",
        description="Data subjects have rights to access, rectification, erasure, and portability",
        patterns=[
            re.compile(r"(?i)(right\s+to\s+(access|erasure|rectif|port|object|forget))"),
        ],
        violation_indicators=[
            "denied",
            "rejected",
            "not possible",
            "cannot comply",
        ],
        severity=FindingSeverity.HIGH,
        remediation="Implement processes to handle data subject requests within 30 days",
    ),
    RegulatoryRequirement(
        framework=RegulatoryFramework.GDPR,
        control_id="GDPR-33",
        title="Breach Notification",
        description="Data breaches must be reported within 72 hours",
        patterns=[
            re.compile(r"(?i)(data\s+breach|security\s+incident|unauthorized\s+access)"),
        ],
        violation_indicators=[
            "delay",
            "not reported",
            "concealed",
            "hidden",
        ],
        severity=FindingSeverity.CRITICAL,
        remediation="Notify supervisory authority within 72 hours of becoming aware of breach",
    ),
    RegulatoryRequirement(
        framework=RegulatoryFramework.GDPR,
        control_id="GDPR-44-49",
        title="International Data Transfers",
        description="Cross-border data transfers require adequate safeguards",
        patterns=[
            re.compile(r"(?i)(transfer|export|cross[\s-]?border|third\s+country)"),
        ],
        violation_indicators=[
            "without safeguards",
            "no adequacy decision",
            "unauthorized transfer",
        ],
        severity=FindingSeverity.HIGH,
        remediation="Implement SCCs, BCRs, or ensure adequacy decision for data transfers",
    ),
]

# PCI-DSS Requirements
PCI_DSS_REQUIREMENTS = [
    RegulatoryRequirement(
        framework=RegulatoryFramework.PCI_DSS,
        control_id="PCI-3.2",
        title="Cardholder Data Storage",
        description="Sensitive authentication data must not be stored after authorization",
        patterns=[
            re.compile(r"(?i)(card\s*number|pan|cvv|cvc|magnetic\s+stripe|track\s+data)"),
        ],
        violation_indicators=[
            "stored",
            "saved",
            "logged",
            "retained",
            "database",
        ],
        severity=FindingSeverity.CRITICAL,
        remediation="Never store CVV, magnetic stripe, or PIN data after authorization",
    ),
    RegulatoryRequirement(
        framework=RegulatoryFramework.PCI_DSS,
        control_id="PCI-3.4",
        title="PAN Display Masking",
        description="PAN must be masked when displayed",
        patterns=[
            re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),  # Unmasked card number
        ],
        violation_indicators=[],  # Pattern itself indicates violation
        severity=FindingSeverity.HIGH,
        remediation="Mask PAN to show only first 6 and last 4 digits",
    ),
    RegulatoryRequirement(
        framework=RegulatoryFramework.PCI_DSS,
        control_id="PCI-8.2",
        title="Strong Authentication",
        description="Strong cryptography must be used for authentication credentials",
        patterns=[
            re.compile(r"(?i)(password|credential|authentication)"),
        ],
        violation_indicators=[
            "plain text",
            "unencrypted",
            "md5",
            "sha1",
            "weak",
        ],
        severity=FindingSeverity.HIGH,
        remediation="Use strong cryptographic hashing (bcrypt, scrypt, or Argon2) for credentials",
    ),
]


class RegulatoryAuditor(BaseAuditor):
    """
    Audits documents for regulatory compliance across multiple frameworks.

    Supports SOX, GDPR, PCI-DSS, and other regulatory standards.
    """

    @property
    def audit_type_id(self) -> str:
        return "regulatory"

    @property
    def display_name(self) -> str:
        return "Regulatory Compliance"

    @property
    def description(self) -> str:
        return "Detects SOX, GDPR, PCI-DSS, and other regulatory compliance issues"

    @property
    def capabilities(self) -> AuditorCapabilities:
        return AuditorCapabilities(
            supports_chunk_analysis=True,
            supports_cross_document=True,
            requires_llm=True,
            finding_categories=[
                "sox_violation",
                "gdpr_violation",
                "pci_dss_violation",
                "regulatory_gap",
                "control_deficiency",
            ],
            supported_document_types=[
                "policy",
                "procedure",
                "financial_report",
                "audit_report",
                "privacy_policy",
                "contract",
            ],
        )

    def __init__(
        self,
        frameworks: Optional[List[RegulatoryFramework]] = None,
    ):
        """
        Initialize regulatory auditor.

        Args:
            frameworks: List of frameworks to check (default: all)
        """
        self._frameworks = frameworks or list(RegulatoryFramework)
        self._requirements = self._load_requirements()
        self._framework_findings: Dict[RegulatoryFramework, int] = {}

    def _load_requirements(self) -> List[RegulatoryRequirement]:
        """Load applicable requirements based on selected frameworks."""
        requirements = []

        if RegulatoryFramework.SOX in self._frameworks:
            requirements.extend(SOX_REQUIREMENTS)
        if RegulatoryFramework.GDPR in self._frameworks:
            requirements.extend(GDPR_REQUIREMENTS)
        if RegulatoryFramework.PCI_DSS in self._frameworks:
            requirements.extend(PCI_DSS_REQUIREMENTS)

        return requirements

    async def analyze_chunk(
        self,
        chunk: ChunkData,
        context: AuditContext,
    ) -> List[AuditFinding]:
        """Analyze a chunk for regulatory compliance issues."""
        findings = []
        content = chunk.content.lower()

        # Pattern-based requirement checking
        for req in self._requirements:
            # Check if content matches requirement scope
            scope_match = any(p.search(chunk.content) for p in req.patterns)
            if not scope_match:
                continue

            # Check for violation indicators
            for indicator in req.violation_indicators:
                if indicator.lower() in content:
                    finding = context.create_finding(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        title=f"{req.framework.value.upper()}: {req.title}",
                        description=f"{req.description}. Control: {req.control_id}",
                        severity=req.severity,
                        category=f"{req.framework.value}_violation",
                        confidence=0.85,
                        evidence_text=self._extract_evidence(chunk.content, indicator),
                        recommendation=req.remediation,
                        tags=[req.framework.value, req.control_id],
                    )
                    findings.append(finding)

                    # Track findings per framework
                    self._framework_findings[req.framework] = (
                        self._framework_findings.get(req.framework, 0) + 1
                    )
                    break  # One finding per requirement per chunk

        # LLM-based analysis for complex requirements
        if len(chunk.content) > 200:
            llm_findings = await self._llm_regulatory_analysis(chunk, context)
            findings.extend(llm_findings)

        return findings

    def _extract_evidence(self, content: str, indicator: str) -> str:
        """Extract evidence context around an indicator."""
        lower_content = content.lower()
        idx = lower_content.find(indicator.lower())
        if idx == -1:
            return ""

        start = max(0, idx - 100)
        end = min(len(content), idx + len(indicator) + 100)
        return "..." + content[start:end] + "..."

    async def _llm_regulatory_analysis(
        self,
        chunk: ChunkData,
        context: AuditContext,
    ) -> List[AuditFinding]:
        """Use LLM for deeper regulatory analysis."""
        findings = []

        try:
            from aragora.agents.api_agents.anthropic import AnthropicAgent  # type: ignore[attr-defined]

            agent = AnthropicAgent(name="regulatory_analyst", model=context.model)

            frameworks_str = ", ".join(f.value.upper() for f in self._frameworks)

            prompt = f"""Analyze this content for regulatory compliance issues.

Frameworks to check: {frameworks_str}

Content:
{chunk.content[:8000]}

Look for:
1. SOX: Financial control weaknesses, audit trail gaps, certification issues
2. GDPR: Privacy violations, consent issues, data subject rights gaps
3. PCI-DSS: Cardholder data exposure, security control gaps
4. General: Policy violations, documentation gaps, control deficiencies

Report issues as JSON array:
[{{"framework": "SOX|GDPR|PCI_DSS|OTHER", "control_id": "...", "title": "...", "severity": "critical|high|medium|low", "evidence": "...", "recommendation": "..."}}]

If no issues, respond with: []"""

            response = await agent.generate(prompt)

            # Parse response
            import json

            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                try:
                    items = json.loads(json_match.group())
                    for item in items:
                        framework = item.get("framework", "OTHER").lower()
                        finding = context.create_finding(
                            document_id=chunk.document_id,
                            chunk_id=chunk.id,
                            title=f"{framework.upper()}: {item.get('title', 'Compliance Issue')}",
                            description=f"Control: {item.get('control_id', 'N/A')}",
                            severity=FindingSeverity(item.get("severity", "medium").lower()),
                            category=f"{framework}_violation",
                            confidence=0.75,
                            evidence_text=item.get("evidence", ""),
                            recommendation=item.get("recommendation", ""),
                            tags=[framework, item.get("control_id", "")],
                        )
                        findings.append(finding)
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.debug(f"Regulatory LLM analysis skipped: {e}")

        return findings

    async def cross_document_analysis(
        self,
        chunks: Sequence[ChunkData],
        context: AuditContext,
    ) -> List[AuditFinding]:
        """Analyze regulatory compliance across all documents."""
        findings = []

        # Check for framework coverage gaps
        for framework in self._frameworks:
            finding_count = self._framework_findings.get(framework, 0)
            if finding_count > 10:
                finding = context.create_finding(
                    document_id=chunks[0].document_id if chunks else "unknown",
                    title=f"{framework.value.upper()} Compliance Concern",
                    description=f"High volume of {framework.value.upper()} issues detected ({finding_count} findings). Systematic compliance review recommended.",
                    severity=FindingSeverity.HIGH,
                    category="regulatory_gap",
                    confidence=0.7,
                    recommendation=f"Conduct comprehensive {framework.value.upper()} compliance assessment",
                    affected_scope="document_set",
                    tags=[framework.value, "systematic_issue"],
                )
                findings.append(finding)

        return findings

    async def post_audit_hook(
        self,
        findings: List[AuditFinding],
        context: AuditContext,
    ) -> List[AuditFinding]:
        """Post-process findings and reset state."""
        self._framework_findings = {}
        return findings


class GDPRDataMapper:
    """
    Utility class for GDPR data mapping.

    Helps identify personal data and data flows for GDPR compliance.
    """

    # Personal data categories per GDPR
    PERSONAL_DATA_PATTERNS = {
        "name": re.compile(r"(?i)(name|full\s+name|first\s+name|last\s+name|surname)"),
        "address": re.compile(r"(?i)(address|street|city|postal\s+code|zip\s+code)"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "id_number": re.compile(r"(?i)(passport|national\s+id|driver.?s?\s+license|ssn)"),
        "financial": re.compile(r"(?i)(bank\s+account|credit\s+card|iban|swift)"),
        "health": re.compile(r"(?i)(medical|health|diagnosis|treatment|prescription)"),
        "biometric": re.compile(r"(?i)(fingerprint|face\s+recognition|retina|dna)"),
        "location": re.compile(r"(?i)(gps|location|coordinates|latitude|longitude)"),
        "online_id": re.compile(r"(?i)(ip\s+address|cookie|device\s+id|mac\s+address)"),
    }

    # Special categories per GDPR Article 9
    SPECIAL_CATEGORIES = [
        "racial_ethnic_origin",
        "political_opinion",
        "religious_belief",
        "trade_union",
        "genetic_data",
        "biometric_data",
        "health_data",
        "sex_life_orientation",
    ]

    def analyze_data_elements(self, text: str) -> Dict[str, List[tuple[int, int]]]:
        """
        Analyze text for personal data elements.

        Returns:
            Dictionary mapping data category to list of (start, end) positions
        """
        results: Dict[str, List[tuple[int, int]]] = {}

        for category, pattern in self.PERSONAL_DATA_PATTERNS.items():
            matches = list(pattern.finditer(text))
            if matches:
                results[category] = [(m.start(), m.end()) for m in matches]

        return results

    def is_special_category(self, data_type: str) -> bool:
        """Check if data type is a special category under GDPR Article 9."""
        return data_type in self.SPECIAL_CATEGORIES


__all__ = [
    "RegulatoryAuditor",
    "RegulatoryFramework",
    "GDPRDataMapper",
    "SOX_REQUIREMENTS",
    "GDPR_REQUIREMENTS",
    "PCI_DSS_REQUIREMENTS",
]
