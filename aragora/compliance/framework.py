"""
Compliance Framework Manager for Enterprise Multi-Agent Control Plane.

Provides configurable compliance checking for industry-specific regulations:
- Pattern-based rule detection
- Severity-weighted scoring
- Framework-specific recommendations
- Multi-framework aggregation

Each framework defines rules that can detect potential compliance issues
in text content (code, documents, plans, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Pattern


class ComplianceSeverity(Enum):
    """Severity levels for compliance issues."""

    CRITICAL = "critical"  # Must be fixed immediately
    HIGH = "high"  # Serious issue, should be fixed
    MEDIUM = "medium"  # Moderate concern
    LOW = "low"  # Minor issue or best practice
    INFO = "info"  # Informational only


@dataclass
class ComplianceIssue:
    """A detected compliance issue."""

    framework: str
    rule_id: str
    severity: ComplianceSeverity
    description: str
    recommendation: str
    matched_text: str = ""
    line_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "description": self.description,
            "recommendation": self.recommendation,
            "matched_text": self.matched_text[:100] if self.matched_text else "",
            "line_number": self.line_number,
            "metadata": self.metadata,
        }


@dataclass
class ComplianceCheckResult:
    """Result of compliance checking."""

    compliant: bool
    issues: list[ComplianceIssue]
    frameworks_checked: list[str]
    score: float  # 0.0 (fully non-compliant) to 1.0 (fully compliant)
    checked_at: datetime = field(default_factory=datetime.now)

    @property
    def critical_issues(self) -> list[ComplianceIssue]:
        return [i for i in self.issues if i.severity == ComplianceSeverity.CRITICAL]

    @property
    def high_issues(self) -> list[ComplianceIssue]:
        return [i for i in self.issues if i.severity == ComplianceSeverity.HIGH]

    def issues_by_framework(self) -> dict[str, list[ComplianceIssue]]:
        result: dict[str, list[ComplianceIssue]] = {}
        for issue in self.issues:
            if issue.framework not in result:
                result[issue.framework] = []
            result[issue.framework].append(issue)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "compliant": self.compliant,
            "score": self.score,
            "frameworks_checked": self.frameworks_checked,
            "issue_count": len(self.issues),
            "critical_count": len(self.critical_issues),
            "high_count": len(self.high_issues),
            "issues": [i.to_dict() for i in self.issues],
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class ComplianceRule:
    """A single compliance rule."""

    id: str
    framework: str
    name: str
    description: str
    severity: ComplianceSeverity
    pattern: str | None = None  # Regex pattern to match
    keywords: list[str] = field(default_factory=list)  # Keywords to detect
    recommendation: str = ""
    category: str = ""
    references: list[str] = field(default_factory=list)

    # Compiled regex (lazy)
    _compiled_pattern: Pattern | None = None

    def get_pattern(self) -> Pattern | None:
        """Get compiled regex pattern."""
        if self.pattern and self._compiled_pattern is None:
            try:
                self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)
            except re.error:
                return None
        return self._compiled_pattern

    def check(self, content: str) -> list[ComplianceIssue]:
        """Check content against this rule."""
        issues = []
        content_lower = content.lower()

        # Check pattern
        pattern = self.get_pattern()
        if pattern:
            for match in pattern.finditer(content):
                # Find line number
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    ComplianceIssue(
                        framework=self.framework,
                        rule_id=self.id,
                        severity=self.severity,
                        description=self.description,
                        recommendation=self.recommendation,
                        matched_text=match.group(0),
                        line_number=line_num,
                        metadata={"category": self.category},
                    )
                )

        # Check keywords
        for keyword in self.keywords:
            if keyword.lower() in content_lower:
                # Find first occurrence
                idx = content_lower.find(keyword.lower())
                keyword_line_num = content[:idx].count("\n") + 1 if idx >= 0 else None
                issues.append(
                    ComplianceIssue(
                        framework=self.framework,
                        rule_id=self.id,
                        severity=self.severity,
                        description=f"{self.description} (keyword: {keyword})",
                        recommendation=self.recommendation,
                        matched_text=keyword,
                        line_number=keyword_line_num,
                        metadata={"category": self.category, "keyword": keyword},
                    )
                )
                break  # Only report once per keyword set

        return issues


@dataclass
class ComplianceFramework:
    """A compliance framework with its rules."""

    id: str
    name: str
    description: str
    version: str
    category: str  # e.g., "security", "privacy", "financial"
    rules: list[ComplianceRule] = field(default_factory=list)
    applicable_verticals: list[str] = field(default_factory=list)

    def check(self, content: str) -> list[ComplianceIssue]:
        """Check content against all rules in this framework."""
        issues = []
        for rule in self.rules:
            issues.extend(rule.check(content))
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "rule_count": len(self.rules),
            "applicable_verticals": self.applicable_verticals,
        }


# =============================================================================
# Pre-defined Compliance Frameworks
# =============================================================================

# HIPAA Framework
HIPAA_FRAMEWORK = ComplianceFramework(
    id="hipaa",
    name="HIPAA",
    description="Health Insurance Portability and Accountability Act",
    version="2013",
    category="healthcare",
    applicable_verticals=["healthcare"],
    rules=[
        ComplianceRule(
            id="hipaa-phi-exposure",
            framework="hipaa",
            name="PHI Exposure Risk",
            description="Potential exposure of Protected Health Information (PHI)",
            severity=ComplianceSeverity.CRITICAL,
            keywords=[
                "patient name",
                "medical record",
                "diagnosis",
                "treatment plan",
                "ssn",
                "social security",
            ],
            recommendation="Ensure PHI is encrypted at rest and in transit. Apply minimum necessary standard.",
            category="privacy",
        ),
        ComplianceRule(
            id="hipaa-unencrypted",
            framework="hipaa",
            name="Unencrypted PHI",
            description="PHI may be stored or transmitted without encryption",
            severity=ComplianceSeverity.CRITICAL,
            pattern=r"(plaintext|unencrypted|clear.?text).{0,50}(patient|health|medical|phi)",
            recommendation="Use AES-256 encryption for PHI at rest and TLS 1.2+ in transit.",
            category="security",
        ),
        ComplianceRule(
            id="hipaa-access-control",
            framework="hipaa",
            name="Access Control Gap",
            description="Insufficient access controls for PHI",
            severity=ComplianceSeverity.HIGH,
            keywords=[
                "public access",
                "no authentication",
                "anonymous access",
                "shared credentials",
            ],
            recommendation="Implement role-based access control (RBAC) with unique user IDs.",
            category="access",
        ),
        ComplianceRule(
            id="hipaa-audit-trail",
            framework="hipaa",
            name="Audit Trail Missing",
            description="PHI access may not be properly logged",
            severity=ComplianceSeverity.HIGH,
            pattern=r"(no|disable|skip).{0,20}(audit|log|track)",
            recommendation="Implement comprehensive audit logging for all PHI access.",
            category="audit",
        ),
        ComplianceRule(
            id="hipaa-minimum-necessary",
            framework="hipaa",
            name="Minimum Necessary Violation",
            description="More PHI may be accessed than necessary",
            severity=ComplianceSeverity.MEDIUM,
            keywords=["all patient data", "full medical record", "entire database", "bulk export"],
            recommendation="Apply minimum necessary standard - only access PHI needed for the task.",
            category="privacy",
        ),
    ],
)

# GDPR Framework
GDPR_FRAMEWORK = ComplianceFramework(
    id="gdpr",
    name="GDPR",
    description="General Data Protection Regulation",
    version="2018",
    category="privacy",
    applicable_verticals=["legal", "healthcare", "software"],
    rules=[
        ComplianceRule(
            id="gdpr-consent",
            framework="gdpr",
            name="Consent Requirement",
            description="Processing personal data without clear consent mechanism",
            severity=ComplianceSeverity.HIGH,
            keywords=["without consent", "implicit consent", "pre-checked", "opt-out only"],
            recommendation="Obtain explicit, informed consent before processing personal data.",
            category="consent",
        ),
        ComplianceRule(
            id="gdpr-data-retention",
            framework="gdpr",
            name="Data Retention Issue",
            description="Personal data may be retained longer than necessary",
            severity=ComplianceSeverity.MEDIUM,
            keywords=["indefinitely", "forever", "no expiration", "permanent storage"],
            recommendation="Implement data retention policies with automatic deletion.",
            category="retention",
        ),
        ComplianceRule(
            id="gdpr-data-transfer",
            framework="gdpr",
            name="Cross-Border Transfer",
            description="Personal data may be transferred outside EEA without safeguards",
            severity=ComplianceSeverity.HIGH,
            pattern=r"(transfer|send|store).{0,30}(outside|non.?eu|third.?country|us|china)",
            recommendation="Ensure Standard Contractual Clauses (SCCs) or adequacy decision.",
            category="transfer",
        ),
        ComplianceRule(
            id="gdpr-subject-rights",
            framework="gdpr",
            name="Data Subject Rights",
            description="Data subject rights may not be supported",
            severity=ComplianceSeverity.MEDIUM,
            keywords=["no deletion", "cannot delete", "no export", "no access request"],
            recommendation="Implement mechanisms for access, rectification, erasure, and portability.",
            category="rights",
        ),
        ComplianceRule(
            id="gdpr-breach-notification",
            framework="gdpr",
            name="Breach Notification",
            description="Breach notification process may be inadequate",
            severity=ComplianceSeverity.HIGH,
            keywords=["no breach notification", "delayed notification", "hide breach"],
            recommendation="Notify supervisory authority within 72 hours of becoming aware of breach.",
            category="breach",
        ),
    ],
)

# SOX Framework
SOX_FRAMEWORK = ComplianceFramework(
    id="sox",
    name="SOX",
    description="Sarbanes-Oxley Act",
    version="2002",
    category="financial",
    applicable_verticals=["accounting", "legal"],
    rules=[
        ComplianceRule(
            id="sox-segregation",
            framework="sox",
            name="Segregation of Duties",
            description="Single person may have excessive control",
            severity=ComplianceSeverity.HIGH,
            keywords=["single approver", "self-approve", "bypass approval", "no separation"],
            recommendation="Implement segregation of duties - separate authorization, custody, and recording.",
            category="control",
        ),
        ComplianceRule(
            id="sox-audit-trail",
            framework="sox",
            name="Audit Trail Requirement",
            description="Financial transactions may not be properly logged",
            severity=ComplianceSeverity.CRITICAL,
            pattern=r"(no|disable|skip|delete).{0,20}(audit|log|trail|history)",
            recommendation="Maintain immutable audit trails for all financial transactions.",
            category="audit",
        ),
        ComplianceRule(
            id="sox-access-control",
            framework="sox",
            name="Access Control Weakness",
            description="Unauthorized access to financial systems possible",
            severity=ComplianceSeverity.HIGH,
            keywords=["shared password", "no mfa", "public access", "guest access"],
            recommendation="Implement strong access controls with MFA for financial systems.",
            category="access",
        ),
        ComplianceRule(
            id="sox-data-integrity",
            framework="sox",
            name="Data Integrity Risk",
            description="Financial data integrity may be compromised",
            severity=ComplianceSeverity.CRITICAL,
            keywords=[
                "manual override",
                "bypass validation",
                "direct database edit",
                "no reconciliation",
            ],
            recommendation="Implement data validation, checksums, and regular reconciliation.",
            category="integrity",
        ),
    ],
)

# OWASP Framework
OWASP_FRAMEWORK = ComplianceFramework(
    id="owasp",
    name="OWASP Top 10",
    description="Open Web Application Security Project Top 10",
    version="2021",
    category="security",
    applicable_verticals=["software"],
    rules=[
        ComplianceRule(
            id="owasp-injection",
            framework="owasp",
            name="Injection Vulnerability",
            description="Potential SQL, command, or code injection vulnerability",
            severity=ComplianceSeverity.CRITICAL,
            pattern=r"(exec|execute|eval|query).{0,30}(\+|format|%s|\$\{|f\")",
            recommendation="Use parameterized queries, prepared statements, or ORMs.",
            category="A03:2021-Injection",
        ),
        ComplianceRule(
            id="owasp-auth",
            framework="owasp",
            name="Broken Authentication",
            description="Authentication mechanism may be vulnerable",
            severity=ComplianceSeverity.HIGH,
            keywords=["plaintext password", "md5 hash", "no rate limit", "session fixation"],
            recommendation="Use strong hashing (bcrypt), implement MFA, and session management.",
            category="A07:2021-Identification",
        ),
        ComplianceRule(
            id="owasp-xss",
            framework="owasp",
            name="Cross-Site Scripting (XSS)",
            description="User input may be rendered without sanitization",
            severity=ComplianceSeverity.HIGH,
            pattern=r"(innerHTML|document\.write|v-html|dangerouslySetInnerHTML)",
            recommendation="Sanitize and encode all user input before rendering.",
            category="A03:2021-Injection",
        ),
        ComplianceRule(
            id="owasp-sensitive-data",
            framework="owasp",
            name="Sensitive Data Exposure",
            description="Sensitive data may be exposed",
            severity=ComplianceSeverity.HIGH,
            pattern=r"(api.?key|password|secret|token|credential).{0,10}=.{0,50}['\"]",
            recommendation="Never hardcode secrets. Use environment variables or secret managers.",
            category="A02:2021-Cryptographic",
        ),
        ComplianceRule(
            id="owasp-security-misconfiguration",
            framework="owasp",
            name="Security Misconfiguration",
            description="Security settings may be misconfigured",
            severity=ComplianceSeverity.MEDIUM,
            keywords=["debug=true", "disable security", "allow all origins", "cors *"],
            recommendation="Follow security hardening guides. Disable debug in production.",
            category="A05:2021-Misconfiguration",
        ),
    ],
)

# PCI-DSS Framework
PCI_DSS_FRAMEWORK = ComplianceFramework(
    id="pci_dss",
    name="PCI-DSS",
    description="Payment Card Industry Data Security Standard",
    version="4.0",
    category="financial",
    applicable_verticals=["software", "accounting"],
    rules=[
        ComplianceRule(
            id="pci-cardholder-data",
            framework="pci_dss",
            name="Cardholder Data Exposure",
            description="Cardholder data may be exposed",
            severity=ComplianceSeverity.CRITICAL,
            keywords=["credit card", "card number", "cvv", "pan", "cardholder"],
            recommendation="Encrypt cardholder data, use tokenization where possible.",
            category="Requirement 3",
        ),
        ComplianceRule(
            id="pci-encryption",
            framework="pci_dss",
            name="Encryption Weakness",
            description="Encryption may not meet PCI-DSS requirements",
            severity=ComplianceSeverity.HIGH,
            pattern=r"(des|rc4|md5|sha1).{0,20}(encrypt|hash|cipher)",
            recommendation="Use AES-256 for encryption, TLS 1.2+ for transmission.",
            category="Requirement 4",
        ),
        ComplianceRule(
            id="pci-access",
            framework="pci_dss",
            name="Access Control Issue",
            description="Access to cardholder data may not be restricted",
            severity=ComplianceSeverity.HIGH,
            keywords=["unrestricted access", "no need to know", "bulk access"],
            recommendation="Implement need-to-know access controls for cardholder data.",
            category="Requirement 7",
        ),
    ],
)

# FDA 21 CFR Part 11 Framework
FDA_21_CFR_FRAMEWORK = ComplianceFramework(
    id="fda_21_cfr",
    name="FDA 21 CFR Part 11",
    description="FDA Electronic Records and Signatures",
    version="2003",
    category="healthcare",
    applicable_verticals=["healthcare"],
    rules=[
        ComplianceRule(
            id="fda-electronic-signature",
            framework="fda_21_cfr",
            name="Electronic Signature Requirements",
            description="Electronic signatures may not meet FDA requirements",
            severity=ComplianceSeverity.HIGH,
            keywords=["unsigned", "no signature", "auto-sign", "batch signature"],
            recommendation="Ensure electronic signatures are unique, linked to record, and auditable.",
            category="Subpart C",
        ),
        ComplianceRule(
            id="fda-audit-trail",
            framework="fda_21_cfr",
            name="Audit Trail Requirements",
            description="Audit trail may not meet ALCOA+ standards",
            severity=ComplianceSeverity.CRITICAL,
            pattern=r"(no|disable|overwrite|delete).{0,20}(audit|log|history|trail)",
            recommendation="Maintain ALCOA+ compliant audit trails (Attributable, Legible, Contemporaneous, Original, Accurate).",
            category="Section 11.10(e)",
        ),
        ComplianceRule(
            id="fda-system-validation",
            framework="fda_21_cfr",
            name="System Validation",
            description="System may not be validated per FDA requirements",
            severity=ComplianceSeverity.MEDIUM,
            keywords=["not validated", "skip validation", "no iq", "no oq", "no pq"],
            recommendation="Complete IQ/OQ/PQ validation with documented evidence.",
            category="Section 11.10(a)",
        ),
    ],
)

# ISO 27001 Framework
ISO_27001_FRAMEWORK = ComplianceFramework(
    id="iso_27001",
    name="ISO 27001",
    description="Information Security Management System",
    version="2022",
    category="security",
    applicable_verticals=["software", "legal"],
    rules=[
        ComplianceRule(
            id="iso-risk-assessment",
            framework="iso_27001",
            name="Risk Assessment Gap",
            description="Risk assessment may be incomplete",
            severity=ComplianceSeverity.MEDIUM,
            keywords=["no risk assessment", "skip risk", "accept all risks"],
            recommendation="Conduct regular risk assessments and maintain risk register.",
            category="A.8.2",
        ),
        ComplianceRule(
            id="iso-access-control",
            framework="iso_27001",
            name="Access Control Policy",
            description="Access control policy may be inadequate",
            severity=ComplianceSeverity.HIGH,
            keywords=["no access policy", "unrestricted", "everyone can access"],
            recommendation="Implement access control policy based on business requirements.",
            category="A.9",
        ),
        ComplianceRule(
            id="iso-incident-response",
            framework="iso_27001",
            name="Incident Response",
            description="Incident response process may be inadequate",
            severity=ComplianceSeverity.MEDIUM,
            keywords=["no incident process", "ad hoc response", "no escalation"],
            recommendation="Establish and test incident response procedures.",
            category="A.16",
        ),
    ],
)

# FedRAMP Framework (NIST 800-53 Moderate Baseline)
FEDRAMP_FRAMEWORK = ComplianceFramework(
    id="fedramp",
    name="FedRAMP",
    description="Federal Risk and Authorization Management Program (NIST 800-53 Moderate)",
    version="Rev 5",
    category="government",
    applicable_verticals=["government", "software"],
    rules=[
        # Access Control (AC) Family
        ComplianceRule(
            id="fedramp-ac-2",
            framework="fedramp",
            name="Account Management",
            description="User account lifecycle may not be properly managed",
            severity=ComplianceSeverity.HIGH,
            keywords=[
                "shared account",
                "generic user",
                "no account review",
                "orphan account",
                "dormant account",
            ],
            recommendation="Implement account lifecycle management with regular reviews per AC-2.",
            category="AC-2",
            references=["NIST 800-53 AC-2"],
        ),
        ComplianceRule(
            id="fedramp-ac-3",
            framework="fedramp",
            name="Access Enforcement",
            description="Access control may not be properly enforced",
            severity=ComplianceSeverity.HIGH,
            keywords=[
                "bypass authorization",
                "no access check",
                "unrestricted access",
                "admin by default",
            ],
            recommendation="Enforce approved authorizations per AC-3 using RBAC/ABAC.",
            category="AC-3",
            references=["NIST 800-53 AC-3"],
        ),
        ComplianceRule(
            id="fedramp-ac-6",
            framework="fedramp",
            name="Least Privilege",
            description="Users may have excessive privileges",
            severity=ComplianceSeverity.HIGH,
            keywords=[
                "full access",
                "admin rights",
                "superuser",
                "root access",
                "elevated privileges",
            ],
            recommendation="Implement least privilege principle per AC-6. Grant only required access.",
            category="AC-6",
            references=["NIST 800-53 AC-6"],
        ),
        ComplianceRule(
            id="fedramp-ac-17",
            framework="fedramp",
            name="Remote Access",
            description="Remote access may not be properly controlled",
            severity=ComplianceSeverity.HIGH,
            pattern=r"(remote|vpn|ssh).{0,30}(no.?auth|anonymous|public|unencrypted)",
            recommendation="Implement multi-factor authentication for remote access per AC-17.",
            category="AC-17",
            references=["NIST 800-53 AC-17"],
        ),
        # Audit and Accountability (AU) Family
        ComplianceRule(
            id="fedramp-au-2",
            framework="fedramp",
            name="Audit Events",
            description="Required audit events may not be captured",
            severity=ComplianceSeverity.CRITICAL,
            pattern=r"(no|disable|skip|delete).{0,20}(audit|log|event)",
            recommendation="Log all required events per AU-2: logins, access, changes, failures.",
            category="AU-2",
            references=["NIST 800-53 AU-2"],
        ),
        ComplianceRule(
            id="fedramp-au-3",
            framework="fedramp",
            name="Content of Audit Records",
            description="Audit records may lack required content",
            severity=ComplianceSeverity.HIGH,
            keywords=["no timestamp", "no user id", "no source", "minimal logging"],
            recommendation="Include what/when/where/who/outcome in all audit records per AU-3.",
            category="AU-3",
            references=["NIST 800-53 AU-3"],
        ),
        ComplianceRule(
            id="fedramp-au-9",
            framework="fedramp",
            name="Protection of Audit Information",
            description="Audit logs may not be protected from tampering",
            severity=ComplianceSeverity.CRITICAL,
            keywords=["delete logs", "modify audit", "overwrite logs", "clear history"],
            recommendation="Protect audit logs from unauthorized modification per AU-9.",
            category="AU-9",
            references=["NIST 800-53 AU-9"],
        ),
        # Configuration Management (CM) Family
        ComplianceRule(
            id="fedramp-cm-2",
            framework="fedramp",
            name="Baseline Configuration",
            description="System may deviate from baseline configuration",
            severity=ComplianceSeverity.MEDIUM,
            keywords=["no baseline", "undocumented config", "ad hoc configuration"],
            recommendation="Maintain and enforce documented baseline configurations per CM-2.",
            category="CM-2",
            references=["NIST 800-53 CM-2"],
        ),
        ComplianceRule(
            id="fedramp-cm-7",
            framework="fedramp",
            name="Least Functionality",
            description="Unnecessary functions/services may be enabled",
            severity=ComplianceSeverity.MEDIUM,
            keywords=[
                "unused service",
                "debug enabled",
                "development mode",
                "test mode in production",
            ],
            recommendation="Disable unnecessary functions and services per CM-7.",
            category="CM-7",
            references=["NIST 800-53 CM-7"],
        ),
        # Identification and Authentication (IA) Family
        ComplianceRule(
            id="fedramp-ia-2",
            framework="fedramp",
            name="Identification and Authentication",
            description="Multi-factor authentication may not be implemented",
            severity=ComplianceSeverity.CRITICAL,
            keywords=["no mfa", "single factor", "password only", "no 2fa"],
            recommendation="Implement multi-factor authentication for privileged and network access per IA-2.",
            category="IA-2",
            references=["NIST 800-53 IA-2"],
        ),
        ComplianceRule(
            id="fedramp-ia-5",
            framework="fedramp",
            name="Authenticator Management",
            description="Credentials may not be properly managed",
            severity=ComplianceSeverity.HIGH,
            keywords=[
                "hardcoded password",
                "default password",
                "weak password",
                "shared credentials",
            ],
            recommendation="Implement secure credential management per IA-5. No hardcoded or default creds.",
            category="IA-5",
            references=["NIST 800-53 IA-5"],
        ),
        # Incident Response (IR) Family
        ComplianceRule(
            id="fedramp-ir-4",
            framework="fedramp",
            name="Incident Handling",
            description="Incident response procedures may be inadequate",
            severity=ComplianceSeverity.HIGH,
            keywords=["no incident plan", "ad hoc response", "no escalation path"],
            recommendation="Implement incident handling capability with documented procedures per IR-4.",
            category="IR-4",
            references=["NIST 800-53 IR-4"],
        ),
        ComplianceRule(
            id="fedramp-ir-6",
            framework="fedramp",
            name="Incident Reporting",
            description="Security incidents may not be reported timely",
            severity=ComplianceSeverity.HIGH,
            keywords=["no incident reporting", "delayed notification", "unreported incident"],
            recommendation="Report incidents to US-CERT within required timeframes per IR-6.",
            category="IR-6",
            references=["NIST 800-53 IR-6"],
        ),
        # System and Communications Protection (SC) Family
        ComplianceRule(
            id="fedramp-sc-8",
            framework="fedramp",
            name="Transmission Confidentiality",
            description="Data in transit may not be encrypted",
            severity=ComplianceSeverity.CRITICAL,
            pattern=r"(http:|unencrypted|plaintext).{0,30}(transmit|send|transfer)",
            recommendation="Encrypt all data in transit using TLS 1.2+ per SC-8.",
            category="SC-8",
            references=["NIST 800-53 SC-8"],
        ),
        ComplianceRule(
            id="fedramp-sc-12",
            framework="fedramp",
            name="Cryptographic Key Management",
            description="Cryptographic keys may not be properly managed",
            severity=ComplianceSeverity.HIGH,
            keywords=["hardcoded key", "embedded key", "no key rotation", "expired certificate"],
            recommendation="Implement FIPS-validated key management per SC-12.",
            category="SC-12",
            references=["NIST 800-53 SC-12"],
        ),
        ComplianceRule(
            id="fedramp-sc-13",
            framework="fedramp",
            name="Cryptographic Protection",
            description="Non-FIPS validated cryptography may be used",
            severity=ComplianceSeverity.HIGH,
            pattern=r"(md5|sha1|des|rc4|3des).{0,20}(encrypt|hash|cipher)",
            recommendation="Use FIPS 140-2 validated cryptographic modules per SC-13.",
            category="SC-13",
            references=["NIST 800-53 SC-13"],
        ),
        ComplianceRule(
            id="fedramp-sc-28",
            framework="fedramp",
            name="Protection of Information at Rest",
            description="Data at rest may not be encrypted",
            severity=ComplianceSeverity.HIGH,
            keywords=["unencrypted storage", "plaintext database", "no disk encryption"],
            recommendation="Encrypt CUI/FCI at rest using FIPS-validated encryption per SC-28.",
            category="SC-28",
            references=["NIST 800-53 SC-28"],
        ),
        # System and Information Integrity (SI) Family
        ComplianceRule(
            id="fedramp-si-2",
            framework="fedramp",
            name="Flaw Remediation",
            description="Security flaws may not be remediated timely",
            severity=ComplianceSeverity.HIGH,
            keywords=["unpatched", "known vulnerability", "outdated software", "cve"],
            recommendation="Remediate high/critical flaws within 30 days per SI-2.",
            category="SI-2",
            references=["NIST 800-53 SI-2"],
        ),
        ComplianceRule(
            id="fedramp-si-3",
            framework="fedramp",
            name="Malicious Code Protection",
            description="Malware protection may be inadequate",
            severity=ComplianceSeverity.HIGH,
            keywords=["no antivirus", "disabled malware scan", "no endpoint protection"],
            recommendation="Implement malicious code protection with automatic updates per SI-3.",
            category="SI-3",
            references=["NIST 800-53 SI-3"],
        ),
        ComplianceRule(
            id="fedramp-si-4",
            framework="fedramp",
            name="System Monitoring",
            description="System monitoring may be inadequate",
            severity=ComplianceSeverity.MEDIUM,
            keywords=["no monitoring", "disabled alerts", "no intrusion detection"],
            recommendation="Implement continuous monitoring with alerting per SI-4.",
            category="SI-4",
            references=["NIST 800-53 SI-4"],
        ),
        ComplianceRule(
            id="fedramp-si-10",
            framework="fedramp",
            name="Information Input Validation",
            description="Input validation may be insufficient",
            severity=ComplianceSeverity.HIGH,
            pattern=r"(no|skip|bypass|disable).{0,20}(validation|sanitiz|input.?check)",
            recommendation="Validate all inputs to prevent injection attacks per SI-10.",
            category="SI-10",
            references=["NIST 800-53 SI-10"],
        ),
    ],
)

# Collect all frameworks
COMPLIANCE_FRAMEWORKS: dict[str, ComplianceFramework] = {
    "hipaa": HIPAA_FRAMEWORK,
    "gdpr": GDPR_FRAMEWORK,
    "sox": SOX_FRAMEWORK,
    "owasp": OWASP_FRAMEWORK,
    "pci_dss": PCI_DSS_FRAMEWORK,
    "fda_21_cfr": FDA_21_CFR_FRAMEWORK,
    "iso_27001": ISO_27001_FRAMEWORK,
    "fedramp": FEDRAMP_FRAMEWORK,
}


class ComplianceFrameworkManager:
    """
    Manages compliance checking across multiple frameworks.

    Provides:
    - Multi-framework content checking
    - Severity-based filtering
    - Framework recommendations by vertical
    - Compliance scoring
    """

    def __init__(
        self,
        frameworks: dict[str, ComplianceFramework] | None = None,
        custom_rules: list[ComplianceRule] | None = None,
    ):
        self._frameworks = frameworks or COMPLIANCE_FRAMEWORKS.copy()
        if custom_rules:
            self._add_custom_rules(custom_rules)

    def _add_custom_rules(self, rules: list[ComplianceRule]) -> None:
        """Add custom rules to existing frameworks."""
        for rule in rules:
            if rule.framework in self._frameworks:
                self._frameworks[rule.framework].rules.append(rule)

    def get_framework(self, framework_id: str) -> ComplianceFramework | None:
        """Get a framework by ID."""
        return self._frameworks.get(framework_id)

    def list_frameworks(self) -> list[dict[str, Any]]:
        """List all available frameworks."""
        return [f.to_dict() for f in self._frameworks.values()]

    def get_frameworks_for_vertical(self, vertical: str) -> list[ComplianceFramework]:
        """Get applicable frameworks for a vertical."""
        return [
            f
            for f in self._frameworks.values()
            if vertical.lower() in [v.lower() for v in f.applicable_verticals]
        ]

    def check(
        self,
        content: str,
        frameworks: list[str] | None = None,
        min_severity: ComplianceSeverity = ComplianceSeverity.LOW,
    ) -> ComplianceCheckResult:
        """
        Check content against specified frameworks.

        Args:
            content: Text content to check
            frameworks: Framework IDs to check (None = all)
            min_severity: Minimum severity to report

        Returns:
            ComplianceCheckResult with issues and score
        """
        if frameworks is None:
            frameworks_to_check = list(self._frameworks.values())
        else:
            frameworks_to_check = [self._frameworks[f] for f in frameworks if f in self._frameworks]

        if not frameworks_to_check:
            return ComplianceCheckResult(
                compliant=True,
                issues=[],
                frameworks_checked=[],
                score=1.0,
            )

        all_issues = []
        for framework in frameworks_to_check:
            issues = framework.check(content)
            # Filter by severity
            severity_order = [
                ComplianceSeverity.CRITICAL,
                ComplianceSeverity.HIGH,
                ComplianceSeverity.MEDIUM,
                ComplianceSeverity.LOW,
                ComplianceSeverity.INFO,
            ]
            min_idx = severity_order.index(min_severity)
            issues = [i for i in issues if severity_order.index(i.severity) <= min_idx]
            all_issues.extend(issues)

        # Calculate compliance score
        score = self._calculate_score(all_issues)

        # Determine overall compliance
        compliant = (
            len(
                [
                    i
                    for i in all_issues
                    if i.severity in (ComplianceSeverity.CRITICAL, ComplianceSeverity.HIGH)
                ]
            )
            == 0
        )

        return ComplianceCheckResult(
            compliant=compliant,
            issues=all_issues,
            frameworks_checked=[f.id for f in frameworks_to_check],
            score=score,
        )

    def _calculate_score(self, issues: list[ComplianceIssue]) -> float:
        """Calculate compliance score from issues."""
        if not issues:
            return 1.0

        # Severity weights
        weights = {
            ComplianceSeverity.CRITICAL: 0.3,
            ComplianceSeverity.HIGH: 0.2,
            ComplianceSeverity.MEDIUM: 0.1,
            ComplianceSeverity.LOW: 0.05,
            ComplianceSeverity.INFO: 0.0,
        }

        total_penalty = sum(weights.get(i.severity, 0) for i in issues)
        # Cap penalty at 1.0
        total_penalty = min(total_penalty, 1.0)

        return 1.0 - total_penalty

    def add_framework(self, framework: ComplianceFramework) -> None:
        """Add a custom framework."""
        self._frameworks[framework.id] = framework

    def add_rule(self, framework_id: str, rule: ComplianceRule) -> bool:
        """Add a rule to an existing framework."""
        if framework_id not in self._frameworks:
            return False
        self._frameworks[framework_id].rules.append(rule)
        return True


async def check_compliance(
    content: str,
    frameworks: list[str] | None = None,
    min_severity: ComplianceSeverity = ComplianceSeverity.LOW,
) -> ComplianceCheckResult:
    """
    Convenience function to check compliance.

    Args:
        content: Text content to check
        frameworks: Framework IDs to check (None = all)
        min_severity: Minimum severity to report

    Returns:
        ComplianceCheckResult with issues and score
    """
    manager = ComplianceFrameworkManager()
    return manager.check(content, frameworks, min_severity)


__all__ = [
    "ComplianceSeverity",
    "ComplianceIssue",
    "ComplianceCheckResult",
    "ComplianceRule",
    "ComplianceFramework",
    "ComplianceFrameworkManager",
    "COMPLIANCE_FRAMEWORKS",
    "check_compliance",
]
