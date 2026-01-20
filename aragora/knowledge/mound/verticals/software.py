"""
Software vertical knowledge module.

Provides domain-specific fact extraction, validation, and pattern detection
for software development artifacts including:
- Security vulnerabilities (OWASP Top 10, CWE)
- Secrets and API keys
- Code quality patterns
- License compliance
- Dependency analysis

Reuses patterns from aragora.audit.audit_types.software.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from aragora.knowledge.mound.verticals.base import (
    BaseVerticalKnowledge,
    ComplianceCheckResult,
    PatternMatch,
    VerticalCapabilities,
    VerticalFact,
)

logger = logging.getLogger(__name__)


@dataclass
class VulnerabilityPattern:
    """Pattern for detecting security vulnerabilities."""

    name: str
    pattern: str
    category: str
    severity: str  # critical, high, medium, low
    cwe: str | None
    description: str
    recommendation: str
    flags: int = re.IGNORECASE | re.MULTILINE


@dataclass
class SecretPattern:
    """Pattern for detecting secrets and API keys."""

    name: str
    pattern: str
    severity: str
    description: str = ""


class SoftwareKnowledge(BaseVerticalKnowledge):
    """
    Software vertical knowledge module.

    Specializes in:
    - Security vulnerability detection (SAST-style)
    - Secret/credential detection
    - Code quality patterns
    - License compliance
    - Dependency analysis

    Reuses patterns from the software audit module for consistency.
    """

    # Vulnerability patterns (subset from software auditor)
    VULNERABILITY_PATTERNS = [
        VulnerabilityPattern(
            name="SQL Injection",
            pattern=r'(?:execute|query|cursor\.execute)\s*\(\s*["\'].*?\%s|'
            r'(?:execute|query)\s*\(\s*f["\']|'
            r'(?:execute|query)\s*\(\s*["\'].*?\+',
            category="injection",
            severity="critical",
            cwe="CWE-89",
            description="Potential SQL injection via string formatting or concatenation",
            recommendation="Use parameterized queries or an ORM",
        ),
        VulnerabilityPattern(
            name="Command Injection",
            pattern=r"(?:os\.system|subprocess\.(?:call|run|Popen)|shell=True)",
            category="injection",
            severity="critical",
            cwe="CWE-78",
            description="Potential command injection via shell execution",
            recommendation="Avoid shell=True, use subprocess with list arguments",
        ),
        VulnerabilityPattern(
            name="XSS",
            pattern=r"innerHTML\s*=|document\.write\s*\(|v-html\s*=|dangerouslySetInnerHTML",
            category="xss",
            severity="high",
            cwe="CWE-79",
            description="Potential cross-site scripting vulnerability",
            recommendation="Sanitize user input before rendering as HTML",
        ),
        VulnerabilityPattern(
            name="Path Traversal",
            pattern=r"(?:open|read|write|Path)\s*\([^)]*\.\./|\.\./",
            category="path_traversal",
            severity="high",
            cwe="CWE-22",
            description="Potential path traversal vulnerability",
            recommendation="Validate and sanitize file paths",
        ),
        VulnerabilityPattern(
            name="Hardcoded Credentials",
            pattern=r'(?:password|passwd|pwd|secret|api_key|apikey|token)\s*=\s*["\'][^"\']+["\']',
            category="secrets",
            severity="high",
            cwe="CWE-798",
            description="Hardcoded credentials detected",
            recommendation="Use environment variables or secure vaults",
        ),
        VulnerabilityPattern(
            name="Weak Cryptography",
            pattern=r"(?:MD5|SHA1|DES|RC4|Blowfish)\s*\(|hashlib\.(?:md5|sha1)",
            category="cryptography",
            severity="medium",
            cwe="CWE-327",
            description="Use of weak cryptographic algorithm",
            recommendation="Use SHA-256 or stronger algorithms",
        ),
        VulnerabilityPattern(
            name="Insecure Random",
            pattern=r"\brandom\.(?:random|randint|choice|shuffle)\b",
            category="cryptography",
            severity="medium",
            cwe="CWE-330",
            description="Use of non-cryptographic random for security",
            recommendation="Use secrets module for security-sensitive operations",
        ),
        VulnerabilityPattern(
            name="SSRF",
            pattern=r"requests\.(?:get|post|put)\s*\([^)]*(?:url|host)",
            category="ssrf",
            severity="high",
            cwe="CWE-918",
            description="Potential server-side request forgery",
            recommendation="Validate and whitelist URLs",
        ),
    ]

    # Secret patterns
    SECRET_PATTERNS = [
        SecretPattern(
            name="AWS Access Key",
            pattern=r"(?:AKIA|A3T|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}",
            severity="critical",
            description="AWS access key ID detected",
        ),
        SecretPattern(
            name="AWS Secret Key",
            pattern=r'(?i)aws.{0,20}[\'"][0-9a-zA-Z/+]{40}[\'"]',
            severity="critical",
            description="AWS secret access key detected",
        ),
        SecretPattern(
            name="GitHub Token",
            pattern=r"gh[ps]_[A-Za-z0-9_]{36}|github_pat_[A-Za-z0-9_]{22,}",
            severity="critical",
            description="GitHub personal access token detected",
        ),
        SecretPattern(
            name="Generic API Key",
            pattern=r'(?i)api[_-]?key[\'"\s:=]+[\'"]?[a-z0-9_-]{20,}[\'"]?',
            severity="high",
            description="Generic API key pattern detected",
        ),
        SecretPattern(
            name="Private Key",
            pattern=r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
            severity="critical",
            description="Private key detected",
        ),
        SecretPattern(
            name="JWT Token",
            pattern=r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
            severity="high",
            description="JSON Web Token detected",
        ),
    ]

    @property
    def vertical_id(self) -> str:
        return "software"

    @property
    def display_name(self) -> str:
        return "Software Development"

    @property
    def description(self) -> str:
        return "Security vulnerabilities, coding patterns, secrets, and best practices"

    @property
    def capabilities(self) -> VerticalCapabilities:
        return VerticalCapabilities(
            supports_pattern_detection=True,
            supports_cross_reference=True,
            supports_compliance_check=True,
            requires_llm=False,
            requires_vector_search=True,
            pattern_categories=[
                "vulnerability",
                "secret",
                "code_quality",
                "license",
                "dependency",
            ],
            compliance_frameworks=["OWASP", "CWE", "SPDX"],
            document_types=["code", "config", "documentation", "dockerfile", "yaml"],
        )

    @property
    def decay_rates(self) -> dict[str, float]:
        """Software-specific decay rates."""
        return {
            "vulnerability": 0.05,  # Security info decays faster
            "secret": 0.1,  # Secrets should be rotated frequently
            "best_practice": 0.01,  # Best practices are stable
            "dependency": 0.03,  # Dependencies change over time
            "license": 0.005,  # Licenses rarely change
            "default": 0.02,
        }

    # -------------------------------------------------------------------------
    # Fact Extraction
    # -------------------------------------------------------------------------

    async def extract_facts(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[VerticalFact]:
        """Extract security-related facts from code content."""
        facts = []
        metadata = metadata or {}

        # Extract vulnerability facts
        for vuln in self.VULNERABILITY_PATTERNS:
            matches = re.findall(vuln.pattern, content, vuln.flags)
            if matches:
                facts.append(
                    self.create_fact(
                        content=f"Potential {vuln.name}: {vuln.description}",
                        category="vulnerability",
                        confidence=0.7,
                        provenance={
                            "pattern": vuln.name,
                            "cwe": vuln.cwe,
                            "match_count": len(matches),
                        },
                        metadata={
                            "severity": vuln.severity,
                            "recommendation": vuln.recommendation,
                            "category": vuln.category,
                            **metadata,
                        },
                    )
                )

        # Extract secret facts
        for secret in self.SECRET_PATTERNS:
            matches = re.findall(secret.pattern, content, re.IGNORECASE)
            if matches:
                # Redact actual secret values
                facts.append(
                    self.create_fact(
                        content=f"Detected {secret.name}: {secret.description}",
                        category="secret",
                        confidence=0.9,  # High confidence for pattern matches
                        provenance={
                            "pattern": secret.name,
                            "match_count": len(matches),
                        },
                        metadata={
                            "severity": secret.severity,
                            **metadata,
                        },
                    )
                )

        return facts

    # -------------------------------------------------------------------------
    # Fact Validation
    # -------------------------------------------------------------------------

    async def validate_fact(
        self,
        fact: VerticalFact,
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, float]:
        """
        Validate a software fact.

        For security facts, validation involves checking if the pattern
        still exists in the codebase and hasn't been fixed.
        """
        if fact.category == "vulnerability":
            # For vulnerabilities, confidence decreases if not re-verified
            # This encourages regular security scans
            new_confidence = max(0.3, fact.confidence * 0.9)
            return True, new_confidence

        if fact.category == "secret":
            # Secrets should be rotated, so confidence drops quickly
            new_confidence = max(0.2, fact.confidence * 0.8)
            return True, new_confidence

        # Default: slight confidence boost for re-validation
        return True, min(0.95, fact.confidence * 1.05)

    # -------------------------------------------------------------------------
    # Pattern Detection
    # -------------------------------------------------------------------------

    async def detect_patterns(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[PatternMatch]:
        """Detect patterns across software facts."""
        patterns = []

        # Group facts by category
        by_category: dict[str, list[VerticalFact]] = {}
        for fact in facts:
            by_category.setdefault(fact.category, []).append(fact)

        # Pattern: Multiple vulnerabilities of same type
        vuln_facts = by_category.get("vulnerability", [])
        if len(vuln_facts) >= 3:
            # Check for recurring vulnerability types
            vuln_types: dict[str, list[str]] = {}
            for fact in vuln_facts:
                vuln_cat = fact.metadata.get("category", "unknown")
                vuln_types.setdefault(vuln_cat, []).append(fact.id)

            for vuln_type, fact_ids in vuln_types.items():
                if len(fact_ids) >= 2:
                    patterns.append(
                        PatternMatch(
                            pattern_id=f"recurring_{vuln_type}_{uuid.uuid4().hex[:8]}",
                            pattern_name=f"Recurring {vuln_type} vulnerabilities",
                            pattern_type="recurring_vulnerability",
                            description=f"Multiple {vuln_type} vulnerabilities detected across codebase",
                            confidence=0.8,
                            supporting_facts=fact_ids,
                            metadata={"vulnerability_type": vuln_type},
                        )
                    )

        # Pattern: Secrets in multiple locations
        secret_facts = by_category.get("secret", [])
        if len(secret_facts) >= 2:
            patterns.append(
                PatternMatch(
                    pattern_id=f"scattered_secrets_{uuid.uuid4().hex[:8]}",
                    pattern_name="Scattered secrets",
                    pattern_type="secret_management",
                    description="Secrets detected in multiple locations - consider centralized secret management",
                    confidence=0.9,
                    supporting_facts=[f.id for f in secret_facts],
                )
            )

        return patterns

    # -------------------------------------------------------------------------
    # Compliance Checking
    # -------------------------------------------------------------------------

    async def check_compliance(
        self,
        facts: Sequence[VerticalFact],
        framework: str,
    ) -> list[ComplianceCheckResult]:
        """Check compliance against security frameworks."""
        results = []

        if framework.upper() == "OWASP":
            results.extend(await self._check_owasp_compliance(facts))
        elif framework.upper() == "CWE":
            results.extend(await self._check_cwe_compliance(facts))

        return results

    async def _check_owasp_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check OWASP Top 10 compliance."""
        results = []

        # Map categories to OWASP Top 10
        owasp_mapping = {
            "injection": "A03:2021 Injection",
            "xss": "A03:2021 Injection",
            "cryptography": "A02:2021 Cryptographic Failures",
            "secrets": "A02:2021 Cryptographic Failures",
            "auth": "A07:2021 Identification and Authentication Failures",
            "access_control": "A01:2021 Broken Access Control",
            "ssrf": "A10:2021 Server-Side Request Forgery",
        }

        # Check each OWASP category
        vuln_facts = [f for f in facts if f.category == "vulnerability"]

        for owasp_id, owasp_name in owasp_mapping.items():
            related_facts = [f for f in vuln_facts if f.metadata.get("category") == owasp_id]

            if related_facts:
                results.append(
                    ComplianceCheckResult(
                        rule_id=f"owasp_{owasp_id}",
                        rule_name=owasp_name,
                        framework="OWASP",
                        passed=False,
                        severity="high" if len(related_facts) > 2 else "medium",
                        findings=[f.content for f in related_facts],
                        evidence=[f.id for f in related_facts],
                        recommendations=[
                            f.metadata.get("recommendation", "Review and fix")
                            for f in related_facts
                        ],
                        confidence=0.8,
                    )
                )

        return results

    async def _check_cwe_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check CWE compliance."""
        results = []

        # Group by CWE
        by_cwe: dict[str, list[VerticalFact]] = {}
        for fact in facts:
            cwe = fact.provenance.get("cwe")
            if cwe:
                by_cwe.setdefault(cwe, []).append(fact)

        for cwe_id, cwe_facts in by_cwe.items():
            results.append(
                ComplianceCheckResult(
                    rule_id=cwe_id,
                    rule_name=f"CWE {cwe_id}",
                    framework="CWE",
                    passed=False,
                    severity=cwe_facts[0].metadata.get("severity", "medium"),
                    findings=[f.content for f in cwe_facts],
                    evidence=[f.id for f in cwe_facts],
                    recommendations=[
                        f.metadata.get("recommendation", "Review and fix") for f in cwe_facts
                    ],
                    confidence=0.85,
                )
            )

        return results
