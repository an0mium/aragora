"""
Security vulnerability detection for document auditing.

Detects:
- Exposed credentials (API keys, passwords, tokens)
- Injection vulnerabilities (SQL, XSS, command injection patterns)
- Insecure configurations
- Data exposure risks
- Hardcoded secrets
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
class SecurityPattern:
    """A pattern for detecting security issues."""

    name: str
    pattern: re.Pattern
    severity: FindingSeverity
    category: str
    description: str
    recommendation: str


class SecurityAuditor:
    """
    Audits documents for security vulnerabilities.

    Combines pattern matching with LLM analysis for
    comprehensive security scanning.
    """

    # Common secret patterns
    SECRET_PATTERNS = [
        SecurityPattern(
            name="AWS Access Key",
            pattern=re.compile(r"AKIA[0-9A-Z]{16}"),
            severity=FindingSeverity.CRITICAL,
            category="exposed_credentials",
            description="AWS Access Key ID detected",
            recommendation="Rotate the AWS credentials immediately and remove from document",
        ),
        SecurityPattern(
            name="AWS Secret Key",
            pattern=re.compile(r"[A-Za-z0-9/+=]{40}(?=.*[A-Z])(?=.*[a-z])(?=.*[0-9])"),
            severity=FindingSeverity.CRITICAL,
            category="exposed_credentials",
            description="Potential AWS Secret Access Key detected",
            recommendation="Rotate AWS credentials and use AWS Secrets Manager",
        ),
        SecurityPattern(
            name="GitHub Token",
            pattern=re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}"),
            severity=FindingSeverity.CRITICAL,
            category="exposed_credentials",
            description="GitHub personal access token detected",
            recommendation="Revoke the token immediately and generate a new one",
        ),
        SecurityPattern(
            name="Slack Token",
            pattern=re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}"),
            severity=FindingSeverity.HIGH,
            category="exposed_credentials",
            description="Slack API token detected",
            recommendation="Revoke the Slack token and regenerate",
        ),
        SecurityPattern(
            name="Generic API Key",
            pattern=re.compile(
                r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?[A-Za-z0-9_\-]{20,}["\']?'
            ),
            severity=FindingSeverity.HIGH,
            category="exposed_credentials",
            description="Generic API key pattern detected",
            recommendation="Remove API key from document and use environment variables",
        ),
        SecurityPattern(
            name="Private Key",
            pattern=re.compile(r"-----BEGIN (RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
            severity=FindingSeverity.CRITICAL,
            category="exposed_credentials",
            description="Private key detected in document",
            recommendation="Never store private keys in documents. Use secure key management",
        ),
        SecurityPattern(
            name="Password in Config",
            pattern=re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?[^\s\'"]{8,}["\']?'),
            severity=FindingSeverity.HIGH,
            category="exposed_credentials",
            description="Hardcoded password detected",
            recommendation="Remove password and use secure credential storage",
        ),
        SecurityPattern(
            name="Database Connection String",
            pattern=re.compile(r'(?i)(mongodb|mysql|postgres|redis|mssql)://[^\s<>"\']+'),
            severity=FindingSeverity.HIGH,
            category="exposed_credentials",
            description="Database connection string with credentials detected",
            recommendation="Use connection string without embedded credentials",
        ),
        SecurityPattern(
            name="JWT Token",
            pattern=re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
            severity=FindingSeverity.MEDIUM,
            category="exposed_credentials",
            description="JWT token detected",
            recommendation="Remove JWT from document. JWTs should be ephemeral",
        ),
    ]

    # Injection patterns
    INJECTION_PATTERNS = [
        SecurityPattern(
            name="SQL Injection Risk",
            pattern=re.compile(
                r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\s+.*\s+(FROM|INTO|WHERE|SET)"
            ),
            severity=FindingSeverity.HIGH,
            category="injection_risk",
            description="SQL query pattern that may be vulnerable to injection",
            recommendation="Use parameterized queries or ORM methods",
        ),
        SecurityPattern(
            name="Command Injection Risk",
            pattern=re.compile(r"(?i)(exec|system|popen|subprocess|eval|`[^`]+`)"),
            severity=FindingSeverity.HIGH,
            category="injection_risk",
            description="Command execution pattern detected",
            recommendation="Validate and sanitize all inputs before command execution",
        ),
        SecurityPattern(
            name="XSS Risk",
            pattern=re.compile(r"(?i)(innerHTML|document\.write|eval\s*\(|<script[^>]*>)"),
            severity=FindingSeverity.MEDIUM,
            category="injection_risk",
            description="Potential XSS vulnerability pattern",
            recommendation="Sanitize user input and use textContent instead of innerHTML",
        ),
    ]

    # Insecure configuration patterns
    CONFIG_PATTERNS = [
        SecurityPattern(
            name="Debug Mode Enabled",
            pattern=re.compile(r"(?i)(debug|DEBUG)\s*[=:]\s*(true|1|yes|on)"),
            severity=FindingSeverity.MEDIUM,
            category="insecure_config",
            description="Debug mode appears to be enabled",
            recommendation="Disable debug mode in production",
        ),
        SecurityPattern(
            name="HTTP Instead of HTTPS",
            pattern=re.compile(r"http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)[\w.-]+"),
            severity=FindingSeverity.MEDIUM,
            category="insecure_config",
            description="Insecure HTTP URL detected (should use HTTPS)",
            recommendation="Use HTTPS for all external connections",
        ),
        SecurityPattern(
            name="Disabled SSL Verification",
            pattern=re.compile(
                r"(?i)(verify\s*[=:]\s*false|ssl_verify\s*[=:]\s*false|CURLOPT_SSL_VERIFYPEER.*false)"
            ),
            severity=FindingSeverity.HIGH,
            category="insecure_config",
            description="SSL certificate verification is disabled",
            recommendation="Enable SSL verification for secure connections",
        ),
        SecurityPattern(
            name="Wildcard CORS",
            pattern=re.compile(
                r'(?i)(Access-Control-Allow-Origin|cors.*origin)\s*[=:]\s*["\']?\*["\']?'
            ),
            severity=FindingSeverity.MEDIUM,
            category="insecure_config",
            description="Wildcard CORS origin allows requests from any domain",
            recommendation="Restrict CORS to specific trusted domains",
        ),
    ]

    def __init__(self):
        """Initialize security auditor."""
        self.all_patterns = self.SECRET_PATTERNS + self.INJECTION_PATTERNS + self.CONFIG_PATTERNS

    async def audit(
        self,
        chunks: list[dict[str, Any]],
        session: AuditSession,
    ) -> list[AuditFinding]:
        """
        Audit document chunks for security issues.

        Args:
            chunks: Document chunks to analyze
            session: Audit session context

        Returns:
            List of security findings
        """
        findings = []

        for chunk in chunks:
            content = chunk.get("content", "")
            chunk_id = chunk.get("id", "")
            document_id = chunk.get("document_id", "")

            # Pattern-based detection
            pattern_findings = self._scan_patterns(
                content,
                session.id,
                document_id,
                chunk_id,
            )
            findings.extend(pattern_findings)

            # LLM-based analysis for complex issues
            if len(content) > 100:
                llm_findings = await self._llm_analysis(
                    content,
                    session.id,
                    document_id,
                    chunk_id,
                    session.model,
                )
                findings.extend(llm_findings)

        return findings

    def _scan_patterns(
        self,
        content: str,
        session_id: str,
        document_id: str,
        chunk_id: str,
    ) -> list[AuditFinding]:
        """Scan content for known security patterns."""
        findings = []

        for pattern in self.all_patterns:
            matches = pattern.pattern.finditer(content)
            for match in matches:
                # Get context around match
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                evidence = content[start:end]

                # Mask sensitive parts
                masked_evidence = self._mask_sensitive(evidence)

                finding = AuditFinding(
                    session_id=session_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    audit_type=AuditType.SECURITY,
                    category=pattern.category,
                    severity=pattern.severity,
                    confidence=0.95,  # High confidence for pattern matches
                    title=pattern.name,
                    description=pattern.description,
                    evidence_text=masked_evidence,
                    evidence_location=f"Position {match.start()}-{match.end()}",
                    recommendation=pattern.recommendation,
                    found_by="security_pattern_scanner",
                )
                findings.append(finding)

        return findings

    async def _llm_analysis(
        self,
        content: str,
        session_id: str,
        document_id: str,
        chunk_id: str,
        model: str,
    ) -> list[AuditFinding]:
        """Use LLM for deeper security analysis."""
        findings = []

        try:
            from aragora.agents.api_agents.gemini import GeminiAgent

            agent = GeminiAgent(name="security_analyst", model=model)

            prompt = f"""Analyze this content for security vulnerabilities:

{content[:10000]}

Look for:
1. Exposed credentials not caught by pattern matching
2. Logic flaws that could lead to security issues
3. Insecure defaults or configurations
4. Information disclosure risks
5. Authentication/authorization issues

Only report issues with HIGH confidence. Format findings as JSON array:
[{{"title": "...", "severity": "critical|high|medium|low", "evidence": "...", "recommendation": "..."}}]

If no issues found, respond with empty array: []"""

            response = await agent.generate(prompt)

            # Parse response
            import json
            import re as regex

            json_match = regex.search(r"\[[\s\S]*\]", response)
            if json_match:
                try:
                    items = json.loads(json_match.group())
                    for item in items:
                        finding = AuditFinding(
                            session_id=session_id,
                            document_id=document_id,
                            chunk_id=chunk_id,
                            audit_type=AuditType.SECURITY,
                            category="llm_detected",
                            severity=FindingSeverity(item.get("severity", "medium").lower()),
                            confidence=0.75,  # Lower confidence for LLM findings
                            title=item.get("title", "Security Issue"),
                            description=item.get("description", ""),
                            evidence_text=item.get("evidence", ""),
                            recommendation=item.get("recommendation", ""),
                            found_by="security_analyst",
                        )
                        findings.append(finding)
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.debug(f"LLM security analysis skipped: {e}")

        return findings

    def _mask_sensitive(self, text: str) -> str:
        """Mask sensitive information in evidence."""
        # Mask API keys
        text = re.sub(
            r"([A-Za-z0-9_\-]{4})[A-Za-z0-9_\-]{16,}([A-Za-z0-9_\-]{4})", r"\1********\2", text
        )
        # Mask passwords
        text = re.sub(
            r'(password|passwd|pwd)\s*[=:]\s*["\']?[^\s\'"]{3}[^\s\'"]*',
            r"\1=***MASKED***",
            text,
            flags=re.IGNORECASE,
        )
        return text


__all__ = ["SecurityAuditor"]
