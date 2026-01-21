"""
Compliance violation detection for document auditing.

Detects violations and risks related to:
- GDPR (data protection, consent, data subject rights)
- HIPAA (PHI handling, access controls)
- SOC 2 (security controls, availability, confidentiality)
- Contractual obligations (SLA violations, obligation mismatches)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from aragora.audit.document_auditor import (
    AuditFinding,
    AuditSession,
    AuditType,
    FindingSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class ComplianceRule:
    """A compliance rule to check against."""

    name: str
    framework: str  # GDPR, HIPAA, SOC2, etc.
    description: str
    severity: FindingSeverity
    check_keywords: list[str]
    required_elements: list[str]
    recommendation: str


class ComplianceAuditor:
    """
    Audits documents for compliance violations.

    Checks against multiple regulatory frameworks and
    identifies missing required elements.
    """

    # GDPR-related rules
    GDPR_RULES = [
        ComplianceRule(
            name="Missing Privacy Policy Reference",
            framework="GDPR",
            description="Document handles personal data but lacks privacy policy reference",
            severity=FindingSeverity.HIGH,
            check_keywords=[
                "personal data",
                "user data",
                "customer information",
                "email address",
                "phone number",
            ],
            required_elements=["privacy policy", "data protection", "GDPR"],
            recommendation="Add reference to privacy policy and ensure GDPR compliance is documented",
        ),
        ComplianceRule(
            name="Missing Data Retention Policy",
            framework="GDPR",
            description="Data collection without specified retention period",
            severity=FindingSeverity.MEDIUM,
            check_keywords=["store", "retain", "keep", "collect", "database"],
            required_elements=["retention period", "data retention", "delete after"],
            recommendation="Specify data retention period as required by GDPR Article 5(1)(e)",
        ),
        ComplianceRule(
            name="Missing Consent Mechanism",
            framework="GDPR",
            description="Processing personal data without documented consent mechanism",
            severity=FindingSeverity.HIGH,
            check_keywords=["collect", "process", "use", "personal data", "PII"],
            required_elements=["consent", "opt-in", "agree", "permission"],
            recommendation="Document consent collection mechanism per GDPR Article 7",
        ),
        ComplianceRule(
            name="Cross-Border Data Transfer",
            framework="GDPR",
            description="International data transfer without adequate safeguards",
            severity=FindingSeverity.HIGH,
            check_keywords=[
                "transfer",
                "international",
                "overseas",
                "cross-border",
                "third country",
            ],
            required_elements=["Standard Contractual Clauses", "adequacy decision", "BCR"],
            recommendation="Document legal basis for international data transfers",
        ),
    ]

    # HIPAA-related rules
    HIPAA_RULES = [
        ComplianceRule(
            name="PHI Without Safeguards",
            framework="HIPAA",
            description="Protected Health Information referenced without security controls",
            severity=FindingSeverity.CRITICAL,
            check_keywords=[
                "patient",
                "medical record",
                "diagnosis",
                "treatment",
                "health information",
                "PHI",
            ],
            required_elements=["encryption", "access control", "audit log"],
            recommendation="Implement required HIPAA safeguards for PHI handling",
        ),
        ComplianceRule(
            name="Missing BAA Reference",
            framework="HIPAA",
            description="Third-party health data sharing without BAA mention",
            severity=FindingSeverity.HIGH,
            check_keywords=[
                "vendor",
                "third party",
                "contractor",
                "service provider",
                "health data",
            ],
            required_elements=["Business Associate Agreement", "BAA", "HIPAA compliance"],
            recommendation="Ensure Business Associate Agreements are in place",
        ),
        ComplianceRule(
            name="Minimum Necessary Violation",
            framework="HIPAA",
            description="Access to PHI beyond minimum necessary",
            severity=FindingSeverity.MEDIUM,
            check_keywords=["all records", "full access", "complete history", "entire database"],
            required_elements=["need to know", "minimum necessary", "role-based"],
            recommendation="Implement minimum necessary access controls",
        ),
    ]

    # SOC 2-related rules
    SOC2_RULES = [
        ComplianceRule(
            name="Missing Access Control Documentation",
            framework="SOC2",
            description="System access without documented access controls",
            severity=FindingSeverity.HIGH,
            check_keywords=["access", "permission", "role", "user", "admin"],
            required_elements=["access control", "authorization", "authentication"],
            recommendation="Document access control policies per SOC 2 CC6.1",
        ),
        ComplianceRule(
            name="Missing Incident Response",
            framework="SOC2",
            description="System handling without incident response procedure",
            severity=FindingSeverity.MEDIUM,
            check_keywords=["alert", "error", "failure", "breach", "incident"],
            required_elements=["incident response", "escalation", "notification"],
            recommendation="Define incident response procedures per SOC 2 CC7.3",
        ),
        ComplianceRule(
            name="Missing Change Management",
            framework="SOC2",
            description="System changes without change management process",
            severity=FindingSeverity.MEDIUM,
            check_keywords=["update", "deploy", "change", "release", "modify"],
            required_elements=["change management", "approval", "testing", "rollback"],
            recommendation="Implement change management per SOC 2 CC8.1",
        ),
    ]

    # PII patterns for detection
    PII_PATTERNS = [
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "email address"),
        (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "phone number"),
        (re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"), "SSN"),
        (re.compile(r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b"), "credit card number"),
        (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "date of birth"),
        (
            re.compile(r"(?i)\b(medical record|patient id|mrn)\s*[#:]?\s*\d+"),
            "medical record number",
        ),
    ]

    def __init__(self):
        """Initialize compliance auditor."""
        self.all_rules = self.GDPR_RULES + self.HIPAA_RULES + self.SOC2_RULES

    async def audit(
        self,
        chunks: list[dict[str, Any]],
        session: AuditSession,
    ) -> list[AuditFinding]:
        """
        Audit document chunks for compliance issues.

        Args:
            chunks: Document chunks to analyze
            session: Audit session context

        Returns:
            List of compliance findings
        """
        findings = []

        # Combine chunks for full document analysis
        full_content = "\n".join(c.get("content", "") for c in chunks)

        # Check for PII presence
        pii_findings = self._detect_pii(full_content, session.id, chunks)
        findings.extend(pii_findings)

        # Check compliance rules
        for rule in self.all_rules:
            rule_findings = self._check_rule(rule, full_content, session.id, chunks)
            findings.extend(rule_findings)

        # LLM-based compliance analysis
        llm_findings = await self._llm_compliance_analysis(
            full_content,
            session.id,
            chunks,
            session.model,
        )
        findings.extend(llm_findings)

        return findings

    def _detect_pii(
        self,
        content: str,
        session_id: str,
        chunks: list[dict[str, Any]],
    ) -> list[AuditFinding]:
        """Detect PII in content."""
        findings = []
        pii_found: dict[str, int] = {}

        for pattern, pii_type in self.PII_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                pii_found[pii_type] = len(matches)

        if pii_found:
            # Check if there's adequate protection mentioned
            has_protection = any(
                term in content.lower()
                for term in ["encrypt", "protect", "secure", "mask", "anonymize", "pseudonymize"]
            )

            severity = FindingSeverity.HIGH if not has_protection else FindingSeverity.MEDIUM

            finding = AuditFinding(
                session_id=session_id,
                document_id=chunks[0].get("document_id", "") if chunks else "",
                audit_type=AuditType.COMPLIANCE,
                category="pii_detected",
                severity=severity,
                confidence=0.9,
                title="Personal Identifiable Information Detected",
                description=f"Found PII types: {', '.join(f'{k} ({v} instances)' for k, v in pii_found.items())}",
                evidence_text="[PII data masked for security]",
                recommendation="Ensure PII is encrypted at rest and in transit. Implement access controls and audit logging.",
                found_by="pii_detector",
                tags=list(pii_found.keys()),
            )
            findings.append(finding)

        return findings

    def _check_rule(
        self,
        rule: ComplianceRule,
        content: str,
        session_id: str,
        chunks: list[dict[str, Any]],
    ) -> list[AuditFinding]:
        """Check a compliance rule against content."""
        findings: list[AuditFinding] = []
        content_lower = content.lower()

        # Check if any trigger keywords are present
        has_trigger = any(keyword.lower() in content_lower for keyword in rule.check_keywords)

        if not has_trigger:
            return findings

        # Check if required elements are present
        missing_elements = [
            elem for elem in rule.required_elements if elem.lower() not in content_lower
        ]

        if missing_elements:
            finding = AuditFinding(
                session_id=session_id,
                document_id=chunks[0].get("document_id", "") if chunks else "",
                audit_type=AuditType.COMPLIANCE,
                category=f"{rule.framework.lower()}_violation",
                severity=rule.severity,
                confidence=0.85,
                title=f"[{rule.framework}] {rule.name}",
                description=rule.description,
                evidence_text=f"Missing required elements: {', '.join(missing_elements)}",
                recommendation=rule.recommendation,
                found_by="compliance_rule_checker",
                tags=[rule.framework],
            )
            findings.append(finding)

        return findings

    async def _llm_compliance_analysis(
        self,
        content: str,
        session_id: str,
        chunks: list[dict[str, Any]],
        model: str,
    ) -> list[AuditFinding]:
        """Use LLM for deeper compliance analysis."""
        findings = []

        try:
            from aragora.agents.api_agents.anthropic import AnthropicAgent  # type: ignore[attr-defined]

            agent = AnthropicAgent(name="compliance_analyst", model="claude-3.5-sonnet")

            prompt = f"""Analyze this document for compliance issues:

{content[:15000]}

Check for violations or risks related to:
1. GDPR - Data protection, consent, data subject rights, international transfers
2. HIPAA - PHI handling, minimum necessary, BAA requirements (if healthcare-related)
3. SOC 2 - Security controls, access management, incident response
4. Contractual - SLA commitments, obligation mismatches, liability clauses

For each issue found, provide:
- Framework (GDPR/HIPAA/SOC2/Contract)
- Specific violation or risk
- Severity (critical/high/medium/low)
- Evidence from the document
- Remediation recommendation

Format as JSON array. If no issues found, respond with empty array: []"""

            response = await agent.generate(prompt)

            # Parse response
            import json
            import re

            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                try:
                    items = json.loads(json_match.group())
                    for item in items:
                        finding = AuditFinding(
                            session_id=session_id,
                            document_id=chunks[0].get("document_id", "") if chunks else "",
                            audit_type=AuditType.COMPLIANCE,
                            category=item.get("framework", "compliance").lower(),
                            severity=FindingSeverity(item.get("severity", "medium").lower()),
                            confidence=0.75,
                            title=item.get("violation", "Compliance Issue"),
                            description=item.get("description", ""),
                            evidence_text=item.get("evidence", ""),
                            recommendation=item.get("remediation", ""),
                            found_by="compliance_analyst",
                            tags=[item.get("framework", "")],
                        )
                        findings.append(finding)
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.debug(f"LLM compliance analysis skipped: {e}")

        return findings


__all__ = ["ComplianceAuditor"]
