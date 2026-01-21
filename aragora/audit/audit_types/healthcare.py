"""
Healthcare document auditing for HIPAA compliance and clinical documentation.

Detects:
- Protected Health Information (PHI) exposure
- HIPAA Privacy Rule violations
- Clinical documentation deficiencies
- Medical coding issues
- Consent and authorization problems
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Sequence

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


@dataclass
class PHIPattern:
    """A pattern for detecting Protected Health Information."""

    name: str
    pattern: re.Pattern[str]
    phi_type: str
    severity: FindingSeverity
    description: str
    hipaa_reference: str


# HIPAA 18 PHI Identifiers
PHI_PATTERNS = [
    PHIPattern(
        name="Social Security Number",
        pattern=re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        phi_type="ssn",
        severity=FindingSeverity.CRITICAL,
        description="Social Security Number detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(A)",
    ),
    PHIPattern(
        name="Medical Record Number",
        pattern=re.compile(r"(?i)(mrn|medical record|patient id)\s*[:#]?\s*\d{5,}"),
        phi_type="medical_record_number",
        severity=FindingSeverity.CRITICAL,
        description="Medical Record Number detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(E)",
    ),
    PHIPattern(
        name="Health Plan Beneficiary Number",
        pattern=re.compile(r"(?i)(beneficiary|member|subscriber)\s*(id|number|#)\s*[:#]?\s*\d{8,}"),
        phi_type="health_plan_id",
        severity=FindingSeverity.CRITICAL,
        description="Health Plan Beneficiary Number detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(F)",
    ),
    PHIPattern(
        name="Account Number",
        pattern=re.compile(r"(?i)(account|acct)\s*(number|#|no)\s*[:#]?\s*\d{6,}"),
        phi_type="account_number",
        severity=FindingSeverity.HIGH,
        description="Account Number detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(G)",
    ),
    PHIPattern(
        name="Date of Birth",
        pattern=re.compile(
            r"(?i)(dob|date of birth|birth\s*date)\s*[:#]?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
        ),
        phi_type="dob",
        severity=FindingSeverity.HIGH,
        description="Date of Birth detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(C)",
    ),
    PHIPattern(
        name="Phone Number",
        pattern=re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        phi_type="phone",
        severity=FindingSeverity.MEDIUM,
        description="Phone number detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(D)",
    ),
    PHIPattern(
        name="Email Address",
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        phi_type="email",
        severity=FindingSeverity.MEDIUM,
        description="Email address detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(D)",
    ),
    PHIPattern(
        name="IP Address",
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        ),
        phi_type="ip_address",
        severity=FindingSeverity.MEDIUM,
        description="IP address detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(O)",
    ),
    PHIPattern(
        name="Device Identifier",
        pattern=re.compile(r"(?i)(device|serial|imei)\s*(id|number|#)\s*[:#]?\s*[A-Z0-9]{8,}"),
        phi_type="device_id",
        severity=FindingSeverity.HIGH,
        description="Device/Serial number detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(I)",
    ),
    PHIPattern(
        name="VIN",
        pattern=re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
        phi_type="vin",
        severity=FindingSeverity.MEDIUM,
        description="Vehicle Identification Number detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(J)",
    ),
    PHIPattern(
        name="URL with PHI",
        pattern=re.compile(r"(?i)(patient|medical|health)[A-Za-z0-9]*\.[a-z]{2,}"),
        phi_type="url",
        severity=FindingSeverity.MEDIUM,
        description="URL potentially containing PHI identifier detected",
        hipaa_reference="45 CFR 164.514(b)(2)(i)(M)",
    ),
]


# Clinical documentation patterns
CLINICAL_PATTERNS = [
    PHIPattern(
        name="Diagnosis Code",
        pattern=re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b"),
        phi_type="icd10",
        severity=FindingSeverity.INFO,
        description="ICD-10 diagnosis code detected",
        hipaa_reference="Clinical Documentation",
    ),
    PHIPattern(
        name="Procedure Code",
        pattern=re.compile(r"\b\d{5}(?:[-\s]?\d{1,2})?\b"),
        phi_type="cpt",
        severity=FindingSeverity.INFO,
        description="CPT procedure code detected",
        hipaa_reference="Clinical Documentation",
    ),
    PHIPattern(
        name="Drug Reference",
        pattern=re.compile(r"(?i)(NDC|rx|prescription)\s*[:#]?\s*\d{4,}[-\d]*"),
        phi_type="ndc",
        severity=FindingSeverity.LOW,
        description="Drug/prescription identifier detected",
        hipaa_reference="Clinical Documentation",
    ),
]


@dataclass
class HIPAAViolation:
    """Represents a potential HIPAA violation."""

    rule: str
    section: str
    description: str
    severity: FindingSeverity
    remediation: str


# Common HIPAA violations to check
HIPAA_VIOLATIONS = [
    HIPAAViolation(
        rule="Privacy Rule - Minimum Necessary",
        section="45 CFR 164.502(b)",
        description="More PHI disclosed than necessary for the purpose",
        severity=FindingSeverity.HIGH,
        remediation="Limit PHI to only what is necessary for the specific purpose",
    ),
    HIPAAViolation(
        rule="Privacy Rule - Patient Rights",
        section="45 CFR 164.524",
        description="Missing or incomplete patient authorization",
        severity=FindingSeverity.HIGH,
        remediation="Ensure proper patient authorization is obtained and documented",
    ),
    HIPAAViolation(
        rule="Security Rule - Access Control",
        section="45 CFR 164.312(a)(1)",
        description="Inadequate access controls for PHI",
        severity=FindingSeverity.HIGH,
        remediation="Implement role-based access control for PHI access",
    ),
    HIPAAViolation(
        rule="Breach Notification Rule",
        section="45 CFR 164.404",
        description="Potential breach of unsecured PHI",
        severity=FindingSeverity.CRITICAL,
        remediation="Follow breach notification procedures within 60 days",
    ),
]


class HealthcareAuditor(BaseAuditor):
    """
    Audits documents for healthcare compliance and PHI exposure.

    Combines pattern matching for PHI detection with LLM analysis
    for clinical documentation quality assessment.
    """

    @property
    def audit_type_id(self) -> str:
        return "healthcare"

    @property
    def display_name(self) -> str:
        return "Healthcare & HIPAA Compliance"

    @property
    def description(self) -> str:
        return "Detects PHI exposure, HIPAA violations, and clinical documentation issues"

    @property
    def capabilities(self) -> AuditorCapabilities:
        return AuditorCapabilities(
            supports_chunk_analysis=True,
            supports_cross_document=True,
            requires_llm=True,
            finding_categories=[
                "phi_exposure",
                "hipaa_violation",
                "clinical_documentation",
                "medical_coding",
                "consent_authorization",
            ],
            supported_document_types=[
                "medical_record",
                "clinical_note",
                "discharge_summary",
                "lab_report",
                "prescription",
                "insurance_claim",
            ],
        )

    def __init__(self):
        """Initialize healthcare auditor."""
        self._phi_patterns = PHI_PATTERNS + CLINICAL_PATTERNS
        self._phi_count: dict[str, int] = {}

    async def analyze_chunk(
        self,
        chunk: ChunkData,
        context: AuditContext,
    ) -> List[AuditFinding]:
        """Analyze a chunk for PHI and HIPAA compliance issues."""
        findings = []

        # PHI Pattern Detection
        phi_findings = self._scan_phi_patterns(chunk, context)
        findings.extend(phi_findings)

        # Track PHI counts for minimum necessary analysis
        for finding in phi_findings:
            phi_type = finding.tags[0] if finding.tags else "unknown"
            self._phi_count[phi_type] = self._phi_count.get(phi_type, 0) + 1

        # Clinical Documentation Analysis (LLM-based)
        if len(chunk.content) > 100:
            clinical_findings = await self._analyze_clinical_documentation(chunk, context)
            findings.extend(clinical_findings)

        return findings

    def _scan_phi_patterns(
        self,
        chunk: ChunkData,
        context: AuditContext,
    ) -> List[AuditFinding]:
        """Scan for PHI patterns."""
        findings = []
        content = chunk.content

        for pattern in self._phi_patterns:
            matches = pattern.pattern.finditer(content)
            for match in matches:
                # Get context around match
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 30)
                evidence = content[start:end]

                # Mask the actual PHI
                masked_evidence = self._mask_phi(evidence, match.group())

                finding = context.create_finding(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    title=f"PHI Detected: {pattern.name}",
                    description=pattern.description,
                    severity=pattern.severity,
                    category="phi_exposure",
                    confidence=0.95,
                    evidence_text=masked_evidence,
                    evidence_location=f"Position {match.start()}-{match.end()}",
                    recommendation=f"De-identify or remove {pattern.phi_type}. Reference: {pattern.hipaa_reference}",
                    tags=[pattern.phi_type, "hipaa", "phi"],
                )
                findings.append(finding)

        return findings

    async def _analyze_clinical_documentation(
        self,
        chunk: ChunkData,
        context: AuditContext,
    ) -> List[AuditFinding]:
        """Analyze clinical documentation quality using LLM."""
        findings = []

        try:
            from aragora.agents.api_agents.anthropic import AnthropicAgent  # type: ignore[attr-defined]

            agent = AnthropicAgent(name="healthcare_analyst", model=context.model)

            prompt = f"""Analyze this clinical documentation for HIPAA compliance and quality:

{chunk.content[:8000]}

Evaluate:
1. Is PHI properly handled and de-identified where required?
2. Is the minimum necessary principle followed?
3. Is patient consent/authorization documented?
4. Is the clinical documentation complete and accurate?
5. Are medical codes (ICD-10, CPT) properly documented?

Report issues as JSON array:
[{{"title": "...", "severity": "critical|high|medium|low", "category": "phi_exposure|hipaa_violation|clinical_documentation|medical_coding|consent_authorization", "evidence": "...", "hipaa_reference": "...", "recommendation": "..."}}]

If no issues, respond with: []"""

            response = await agent.generate(prompt)

            # Parse response
            import json

            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                try:
                    items = json.loads(json_match.group())
                    for item in items:
                        finding = context.create_finding(
                            document_id=chunk.document_id,
                            chunk_id=chunk.id,
                            title=item.get("title", "Healthcare Issue"),
                            description=f"{item.get('description', '')} Reference: {item.get('hipaa_reference', 'N/A')}",
                            severity=FindingSeverity(item.get("severity", "medium").lower()),
                            category=item.get("category", "clinical_documentation"),
                            confidence=0.8,
                            evidence_text=item.get("evidence", ""),
                            recommendation=item.get("recommendation", ""),
                            tags=["hipaa", "clinical"],
                        )
                        findings.append(finding)
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.debug(f"Clinical documentation analysis skipped: {e}")

        return findings

    async def cross_document_analysis(
        self,
        chunks: Sequence[ChunkData],
        context: AuditContext,
    ) -> List[AuditFinding]:
        """Analyze PHI patterns across all documents."""
        findings = []

        # Check for minimum necessary violations (excessive PHI)
        total_phi = sum(self._phi_count.values())
        if total_phi > 50:
            finding = context.create_finding(
                document_id=chunks[0].document_id if chunks else "unknown",
                title="Minimum Necessary Principle Concern",
                description=f"High volume of PHI detected ({total_phi} instances). Review if all PHI is necessary for the intended purpose.",
                severity=FindingSeverity.HIGH,
                category="hipaa_violation",
                confidence=0.7,
                recommendation="Review PHI usage and ensure minimum necessary standard is met per 45 CFR 164.502(b)",
                affected_scope="document_set",
                tags=["hipaa", "minimum_necessary"],
            )
            findings.append(finding)

        # Check for PHI diversity (multiple PHI types may indicate over-collection)
        if len(self._phi_count) > 5:
            finding = context.create_finding(
                document_id=chunks[0].document_id if chunks else "unknown",
                title="PHI Diversity Concern",
                description=f"Multiple PHI types detected ({len(self._phi_count)} types). Consider if all are necessary.",
                severity=FindingSeverity.MEDIUM,
                category="hipaa_violation",
                confidence=0.6,
                recommendation="Review data collection practices and ensure only necessary PHI types are collected",
                affected_scope="document_set",
                tags=["hipaa", "data_minimization"],
            )
            findings.append(finding)

        return findings

    def _mask_phi(self, text: str, phi: str) -> str:
        """Mask PHI in evidence text."""
        if len(phi) <= 4:
            return text.replace(phi, "****")
        return text.replace(phi, phi[:2] + "****" + phi[-2:])

    async def post_audit_hook(
        self,
        findings: List[AuditFinding],
        context: AuditContext,
    ) -> List[AuditFinding]:
        """Post-process findings and generate summary."""
        # Reset PHI count for next audit
        self._phi_count = {}

        # Deduplicate similar findings
        seen = set()
        unique_findings = []
        for finding in findings:
            key = (finding.title, finding.document_id, finding.category)
            if key not in seen:
                seen.add(key)
                unique_findings.append(finding)

        return unique_findings


class PHIDetector:
    """
    Standalone PHI detector for quick scanning.

    Can be used independently of the full audit system.
    """

    def __init__(self):
        self._patterns = PHI_PATTERNS

    def scan(self, text: str) -> List[dict[str, Any]]:
        """
        Scan text for PHI.

        Returns:
            List of detected PHI with type and location
        """
        results = []
        for pattern in self._patterns:
            for match in pattern.pattern.finditer(text):
                results.append(
                    {
                        "type": pattern.phi_type,
                        "name": pattern.name,
                        "start": match.start(),
                        "end": match.end(),
                        "severity": pattern.severity.value,
                        "hipaa_reference": pattern.hipaa_reference,
                    }
                )
        return results

    def redact(self, text: str) -> str:
        """
        Redact all PHI from text.

        Returns:
            Text with PHI replaced by [REDACTED]
        """
        redacted = text
        # Process patterns in order of specificity (longest matches first)
        all_matches = []
        for pattern in self._patterns:
            for match in pattern.pattern.finditer(text):
                all_matches.append((match.start(), match.end(), pattern.name))

        # Sort by start position descending to replace from end
        all_matches.sort(key=lambda x: x[0], reverse=True)

        for start, end, name in all_matches:
            redacted = redacted[:start] + f"[{name} REDACTED]" + redacted[end:]

        return redacted


__all__ = ["HealthcareAuditor", "PHIDetector", "PHI_PATTERNS"]
