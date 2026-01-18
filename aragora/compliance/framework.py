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
from typing import Any, Callable, Dict, List, Optional, Pattern, Set


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
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    issues: List[ComplianceIssue]
    frameworks_checked: List[str]
    score: float  # 0.0 (fully non-compliant) to 1.0 (fully compliant)
    checked_at: datetime = field(default_factory=datetime.now)

    @property
    def critical_issues(self) -> List[ComplianceIssue]:
        return [i for i in self.issues if i.severity == ComplianceSeverity.CRITICAL]

    @property
    def high_issues(self) -> List[ComplianceIssue]:
        return [i for i in self.issues if i.severity == ComplianceSeverity.HIGH]

    def issues_by_framework(self) -> Dict[str, List[ComplianceIssue]]:
        result: Dict[str, List[ComplianceIssue]] = {}
        for issue in self.issues:
            if issue.framework not in result:
                result[issue.framework] = []
            result[issue.framework].append(issue)
        return result

    def to_dict(self) -> Dict[str, Any]:
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
    pattern: Optional[str] = None  # Regex pattern to match
    keywords: List[str] = field(default_factory=list)  # Keywords to detect
    recommendation: str = ""
    category: str = ""
    references: List[str] = field(default_factory=list)

    # Compiled regex (lazy)
    _compiled_pattern: Optional[Pattern] = None

    def get_pattern(self) -> Optional[Pattern]:
        """Get compiled regex pattern."""
        if self.pattern and self._compiled_pattern is None:
            try:
                self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)
            except re.error:
                return None
        return self._compiled_pattern

    def check(self, content: str) -> List[ComplianceIssue]:
        """Check content against this rule."""
        issues = []
        content_lower = content.lower()

        # Check pattern
        pattern = self.get_pattern()
        if pattern:
            for match in pattern.finditer(content):
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                issues.append(ComplianceIssue(
                    framework=self.framework,
                    rule_id=self.id,
                    severity=self.severity,
                    description=self.description,
                    recommendation=self.recommendation,
                    matched_text=match.group(0),
                    line_number=line_num,
                    metadata={"category": self.category},
                ))

        # Check keywords
        for keyword in self.keywords:
            if keyword.lower() in content_lower:
                # Find first occurrence
                idx = content_lower.find(keyword.lower())
                line_num = content[:idx].count('\n') + 1 if idx >= 0 else None
                issues.append(ComplianceIssue(
                    framework=self.framework,
                    rule_id=self.id,
                    severity=self.severity,
                    description=f"{self.description} (keyword: {keyword})",
                    recommendation=self.recommendation,
                    matched_text=keyword,
                    line_number=line_num,
                    metadata={"category": self.category, "keyword": keyword},
                ))
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
    rules: List[ComplianceRule] = field(default_factory=list)
    applicable_verticals: List[str] = field(default_factory=list)

    def check(self, content: str) -> List[ComplianceIssue]:
        """Check content against all rules in this framework."""
        issues = []
        for rule in self.rules:
            issues.extend(rule.check(content))
        return issues

    def to_dict(self) -> Dict[str, Any]:
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
            keywords=["patient name", "medical record", "diagnosis", "treatment plan", "ssn", "social security"],
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
            keywords=["public access", "no authentication", "anonymous access", "shared credentials"],
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
            keywords=["manual override", "bypass validation", "direct database edit", "no reconciliation"],
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

# Collect all frameworks
COMPLIANCE_FRAMEWORKS: Dict[str, ComplianceFramework] = {
    "hipaa": HIPAA_FRAMEWORK,
    "gdpr": GDPR_FRAMEWORK,
    "sox": SOX_FRAMEWORK,
    "owasp": OWASP_FRAMEWORK,
    "pci_dss": PCI_DSS_FRAMEWORK,
    "fda_21_cfr": FDA_21_CFR_FRAMEWORK,
    "iso_27001": ISO_27001_FRAMEWORK,
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
        frameworks: Optional[Dict[str, ComplianceFramework]] = None,
        custom_rules: Optional[List[ComplianceRule]] = None,
    ):
        self._frameworks = frameworks or COMPLIANCE_FRAMEWORKS.copy()
        if custom_rules:
            self._add_custom_rules(custom_rules)

    def _add_custom_rules(self, rules: List[ComplianceRule]) -> None:
        """Add custom rules to existing frameworks."""
        for rule in rules:
            if rule.framework in self._frameworks:
                self._frameworks[rule.framework].rules.append(rule)

    def get_framework(self, framework_id: str) -> Optional[ComplianceFramework]:
        """Get a framework by ID."""
        return self._frameworks.get(framework_id)

    def list_frameworks(self) -> List[Dict[str, Any]]:
        """List all available frameworks."""
        return [f.to_dict() for f in self._frameworks.values()]

    def get_frameworks_for_vertical(self, vertical: str) -> List[ComplianceFramework]:
        """Get applicable frameworks for a vertical."""
        return [
            f for f in self._frameworks.values()
            if vertical.lower() in [v.lower() for v in f.applicable_verticals]
        ]

    def check(
        self,
        content: str,
        frameworks: Optional[List[str]] = None,
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
            frameworks_to_check = [
                self._frameworks[f] for f in frameworks
                if f in self._frameworks
            ]

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
        compliant = len([i for i in all_issues if i.severity in (
            ComplianceSeverity.CRITICAL, ComplianceSeverity.HIGH
        )]) == 0

        return ComplianceCheckResult(
            compliant=compliant,
            issues=all_issues,
            frameworks_checked=[f.id for f in frameworks_to_check],
            score=score,
        )

    def _calculate_score(self, issues: List[ComplianceIssue]) -> float:
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
    frameworks: Optional[List[str]] = None,
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
