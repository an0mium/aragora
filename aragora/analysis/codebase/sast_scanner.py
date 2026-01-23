"""
SAST Scanner - Static Application Security Testing.

Integrates with Semgrep for comprehensive static analysis with OWASP mapping.
Falls back to local pattern matching when Semgrep is not available.

Features:
- OWASP Top 10 vulnerability detection
- CWE ID mapping for findings
- Multi-language support (Python, JavaScript, Go, Java, TypeScript, Ruby)
- Custom rule support
- Severity classification
- False positive filtering via confidence scoring
- Async scanning with progress reporting
- SecurityEventEmitter integration for critical findings

Usage:
    from aragora.analysis.codebase.sast_scanner import SASTScanner

    scanner = SASTScanner()
    await scanner.initialize()

    # Scan a repository
    result = await scanner.scan_repository("/path/to/repo")
    print(f"Found {len(result.findings)} issues")

    # Scan with specific rules
    result = await scanner.scan_with_rules(
        path="/path/to/repo",
        rule_sets=["owasp-top-10", "cwe-top-25"],
    )

    # Get available rulesets
    rulesets = await scanner.get_available_rulesets()

    # Scan with progress reporting
    async def on_progress(current, total, message):
        print(f"[{current}/{total}] {message}")

    result = await scanner.scan_repository("/path/to/repo", progress_callback=on_progress)

Semgrep Installation:
    If Semgrep is not installed, install it with:
        pip install semgrep
    Or:
        brew install semgrep  # macOS
        python3 -m pip install semgrep  # Python
    See: https://semgrep.dev/docs/getting-started/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Callable[[int, int, str], Coroutine[Any, Any, None]]


class SASTSeverity(Enum):
    """Severity levels for SAST findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        """Get numeric severity level for comparison."""
        levels = {"info": 0, "warning": 1, "error": 2, "critical": 3}
        return levels.get(self.value, 0)

    def __ge__(self, other: "SASTSeverity") -> bool:
        """Compare severity levels."""
        if isinstance(other, SASTSeverity):
            return self.level >= other.level
        return NotImplemented

    def __gt__(self, other: "SASTSeverity") -> bool:
        """Compare severity levels."""
        if isinstance(other, SASTSeverity):
            return self.level > other.level
        return NotImplemented

    def __le__(self, other: "SASTSeverity") -> bool:
        """Compare severity levels."""
        if isinstance(other, SASTSeverity):
            return self.level <= other.level
        return NotImplemented

    def __lt__(self, other: "SASTSeverity") -> bool:
        """Compare severity levels."""
        if isinstance(other, SASTSeverity):
            return self.level < other.level
        return NotImplemented


class OWASPCategory(Enum):
    """OWASP Top 10 2021 categories."""

    A01_BROKEN_ACCESS_CONTROL = "A01:2021 - Broken Access Control"
    A02_CRYPTOGRAPHIC_FAILURES = "A02:2021 - Cryptographic Failures"
    A03_INJECTION = "A03:2021 - Injection"
    A04_INSECURE_DESIGN = "A04:2021 - Insecure Design"
    A05_SECURITY_MISCONFIGURATION = "A05:2021 - Security Misconfiguration"
    A06_VULNERABLE_COMPONENTS = "A06:2021 - Vulnerable and Outdated Components"
    A07_AUTH_FAILURES = "A07:2021 - Identification and Authentication Failures"
    A08_DATA_INTEGRITY = "A08:2021 - Software and Data Integrity Failures"
    A09_LOGGING_FAILURES = "A09:2021 - Security Logging and Monitoring Failures"
    A10_SSRF = "A10:2021 - Server-Side Request Forgery"
    UNKNOWN = "Unknown"


@dataclass
class SASTFinding:
    """A finding from SAST analysis."""

    rule_id: str
    file_path: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    message: str
    severity: SASTSeverity
    confidence: float  # 0.0 to 1.0
    language: str
    snippet: str = ""

    # Security metadata
    cwe_ids: List[str] = field(default_factory=list)
    owasp_category: OWASPCategory = OWASPCategory.UNKNOWN
    vulnerability_class: str = ""
    remediation: str = ""

    # Source information
    source: str = "semgrep"  # semgrep, local, custom
    rule_name: str = ""
    rule_url: str = ""

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_false_positive: bool = False
    triaged: bool = False

    # Finding ID for tracking
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "rule_id": self.rule_id,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
            "message": self.message,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "language": self.language,
            "snippet": self.snippet,
            "cwe_ids": self.cwe_ids,
            "owasp_category": self.owasp_category.value,
            "vulnerability_class": self.vulnerability_class,
            "remediation": self.remediation,
            "source": self.source,
            "rule_name": self.rule_name,
            "rule_url": self.rule_url,
            "metadata": self.metadata,
            "is_false_positive": self.is_false_positive,
            "triaged": self.triaged,
        }


@dataclass
class SASTScanResult:
    """Result of a SAST scan."""

    repository_path: str
    scan_id: str
    findings: List[SASTFinding]
    scanned_files: int
    skipped_files: int
    scan_duration_ms: float
    languages_detected: List[str]
    rules_used: List[str]
    errors: List[str] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=datetime.now)

    @property
    def findings_by_severity(self) -> Dict[str, int]:
        """Count findings by severity."""
        counts: Dict[str, int] = {}
        for finding in self.findings:
            sev = finding.severity.value
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    @property
    def findings_by_owasp(self) -> Dict[str, int]:
        """Count findings by OWASP category."""
        counts: Dict[str, int] = {}
        for finding in self.findings:
            cat = finding.owasp_category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repository_path": self.repository_path,
            "scan_id": self.scan_id,
            "findings": [f.to_dict() for f in self.findings],
            "findings_count": len(self.findings),
            "scanned_files": self.scanned_files,
            "skipped_files": self.skipped_files,
            "scan_duration_ms": self.scan_duration_ms,
            "languages_detected": self.languages_detected,
            "rules_used": self.rules_used,
            "errors": self.errors,
            "scanned_at": self.scanned_at.isoformat(),
            "summary": {
                "by_severity": self.findings_by_severity,
                "by_owasp": self.findings_by_owasp,
            },
        }


@dataclass
class SASTConfig:
    """Configuration for SAST scanner."""

    # Semgrep settings
    semgrep_path: str = "semgrep"
    use_semgrep: bool = True
    semgrep_timeout: int = 300  # 5 minutes

    # Rule sets to use
    default_rule_sets: List[str] = field(
        default_factory=lambda: [
            "p/owasp-top-ten",
            "p/security-audit",
        ]
    )

    # Custom rules directory
    custom_rules_dir: Optional[str] = None

    # File filters
    max_file_size_kb: int = 500
    excluded_patterns: List[str] = field(
        default_factory=lambda: [
            "node_modules/",
            "venv/",
            ".git/",
            "__pycache__/",
            "*.min.js",
            "*.bundle.js",
            "vendor/",
            "dist/",
            "build/",
        ]
    )

    # Language settings
    supported_languages: List[str] = field(
        default_factory=lambda: [
            "python",
            "javascript",
            "typescript",
            "go",
            "java",
            "ruby",
            "php",
            "csharp",
        ]
    )

    # Severity filtering
    min_severity: SASTSeverity = SASTSeverity.WARNING

    # Performance
    max_concurrent_files: int = 10

    # False positive filtering
    min_confidence_threshold: float = 0.5  # Filter findings below this confidence
    enable_false_positive_filtering: bool = True

    # Security event integration
    emit_security_events: bool = True
    critical_finding_threshold: int = 1  # Emit event when this many critical findings


# Available Semgrep rulesets with descriptions
AVAILABLE_RULESETS: Dict[str, Dict[str, str]] = {
    # OWASP rulesets
    "p/owasp-top-ten": {
        "name": "OWASP Top 10",
        "description": "Rules covering OWASP Top 10 2021 vulnerabilities",
        "category": "owasp",
    },
    "p/owasp-top-ten-2017": {
        "name": "OWASP Top 10 2017",
        "description": "Rules covering OWASP Top 10 2017 vulnerabilities",
        "category": "owasp",
    },
    # Security rulesets
    "p/security-audit": {
        "name": "Security Audit",
        "description": "Comprehensive security audit rules",
        "category": "security",
    },
    "p/secrets": {
        "name": "Secrets Detection",
        "description": "Detection of hardcoded secrets and credentials",
        "category": "security",
    },
    "p/supply-chain": {
        "name": "Supply Chain",
        "description": "Supply chain security vulnerabilities",
        "category": "security",
    },
    # CWE rulesets
    "p/cwe-top-25": {
        "name": "CWE Top 25",
        "description": "CWE Top 25 Most Dangerous Software Weaknesses",
        "category": "cwe",
    },
    # Language-specific rulesets
    "p/python": {
        "name": "Python Security",
        "description": "Python-specific security rules",
        "category": "language",
    },
    "p/javascript": {
        "name": "JavaScript Security",
        "description": "JavaScript-specific security rules",
        "category": "language",
    },
    "p/typescript": {
        "name": "TypeScript Security",
        "description": "TypeScript-specific security rules",
        "category": "language",
    },
    "p/go": {
        "name": "Go Security",
        "description": "Go-specific security rules",
        "category": "language",
    },
    "p/java": {
        "name": "Java Security",
        "description": "Java-specific security rules",
        "category": "language",
    },
    "p/ruby": {
        "name": "Ruby Security",
        "description": "Ruby-specific security rules",
        "category": "language",
    },
    # Framework-specific
    "p/django": {
        "name": "Django Security",
        "description": "Django framework security rules",
        "category": "framework",
    },
    "p/flask": {
        "name": "Flask Security",
        "description": "Flask framework security rules",
        "category": "framework",
    },
    "p/react": {
        "name": "React Security",
        "description": "React framework security rules",
        "category": "framework",
    },
    "p/nodejs": {
        "name": "Node.js Security",
        "description": "Node.js security rules",
        "category": "framework",
    },
    # Additional rulesets
    "p/insecure-transport": {
        "name": "Insecure Transport",
        "description": "Detection of insecure transport layer configurations",
        "category": "security",
    },
    "p/jwt": {
        "name": "JWT Security",
        "description": "JSON Web Token security vulnerabilities",
        "category": "security",
    },
    "p/sql-injection": {
        "name": "SQL Injection",
        "description": "SQL injection vulnerability detection",
        "category": "injection",
    },
    "p/xss": {
        "name": "XSS",
        "description": "Cross-site scripting vulnerability detection",
        "category": "injection",
    },
    "p/command-injection": {
        "name": "Command Injection",
        "description": "OS command injection vulnerability detection",
        "category": "injection",
    },
}

# Fix recommendations by CWE category
CWE_FIX_RECOMMENDATIONS: Dict[str, str] = {
    # Injection
    "CWE-78": "Use parameterized commands or shell escaping. Avoid shell=True with user input.",
    "CWE-79": "Sanitize user input before rendering. Use templating engines with auto-escaping.",
    "CWE-89": "Use parameterized queries or an ORM. Never concatenate user input in SQL.",
    "CWE-94": "Avoid eval/exec with user input. Use safe alternatives like ast.literal_eval.",
    "CWE-95": "Never use eval() with untrusted data. Parse data using safe methods.",
    # Cryptographic
    "CWE-327": "Use strong cryptographic algorithms (AES-256, SHA-256+). Avoid MD5/SHA1 for security.",
    "CWE-328": "Use bcrypt, scrypt, or Argon2 for password hashing. Never use MD5/SHA1.",
    "CWE-330": "Use cryptographically secure random number generators (secrets module in Python).",
    "CWE-338": "Replace weak PRNG with secrets.token_bytes() or os.urandom().",
    # Authentication
    "CWE-259": "Store credentials in environment variables or secure vaults, not in code.",
    "CWE-287": "Implement proper authentication. Verify credentials on every request.",
    "CWE-306": "Add authentication checks before accessing sensitive functionality.",
    "CWE-798": "Move hardcoded credentials to environment variables or secret management.",
    # Access Control
    "CWE-22": "Validate and sanitize file paths. Use os.path.realpath() and check allowed directories.",
    "CWE-284": "Implement proper access control checks. Follow principle of least privilege.",
    "CWE-352": "Implement CSRF tokens for state-changing operations.",
    "CWE-862": "Add authorization checks before accessing resources.",
    # Data Integrity
    "CWE-502": "Avoid deserializing untrusted data. Use safe serialization formats like JSON.",
    "CWE-494": "Verify integrity of downloaded code using checksums or signatures.",
    # Configuration
    "CWE-16": "Review security configuration. Disable debug mode in production.",
    "CWE-614": "Set Secure flag on cookies containing sensitive data.",
    "CWE-1004": "Set HttpOnly flag on session cookies.",
    # SSRF
    "CWE-918": "Validate and whitelist allowed URLs. Block internal network ranges.",
    # Logging
    "CWE-117": "Sanitize user input before logging to prevent log injection.",
    "CWE-532": "Avoid logging sensitive data. Mask or redact sensitive information.",
}


# CWE to OWASP mapping for common vulnerabilities
CWE_TO_OWASP: Dict[str, OWASPCategory] = {
    # A01: Broken Access Control
    "CWE-22": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Path Traversal
    "CWE-23": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Relative Path Traversal
    "CWE-35": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Path Traversal
    "CWE-59": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Link Following
    "CWE-200": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Information Exposure
    "CWE-201": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Insertion of Sensitive Info
    "CWE-219": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Storage of File with Sensitive Data
    "CWE-264": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Permissions, Privileges
    "CWE-275": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Permission Issues
    "CWE-276": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Incorrect Default Permissions
    "CWE-284": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Improper Access Control
    "CWE-285": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Improper Authorization
    "CWE-352": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # CSRF
    "CWE-359": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Privacy Violation
    "CWE-425": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Direct Request
    "CWE-639": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Authorization Bypass via IDOR
    "CWE-732": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Incorrect Permission Assignment
    "CWE-862": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Missing Authorization
    "CWE-863": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,  # Incorrect Authorization
    # A02: Cryptographic Failures
    "CWE-261": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Weak Encoding
    "CWE-296": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Improper Certificate Validation
    "CWE-310": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Cryptographic Issues
    "CWE-319": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Cleartext Transmission
    "CWE-321": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Hard-coded Cryptographic Key
    "CWE-322": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Key Exchange without Entity Auth
    "CWE-323": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Reusing Nonce
    "CWE-324": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Use of Expired Key
    "CWE-325": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Missing Cryptographic Step
    "CWE-326": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Inadequate Encryption Strength
    "CWE-327": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Broken Crypto Algorithm
    "CWE-328": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Reversible One-Way Hash
    "CWE-329": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Not Using Random IV
    "CWE-330": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Insufficient Randomness
    "CWE-331": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Insufficient Entropy
    "CWE-335": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Incorrect Usage of Seeds
    "CWE-336": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Same Seed in PRNG
    "CWE-337": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Predictable Seed
    "CWE-338": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Weak PRNG
    "CWE-340": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Predictable from Observable State
    "CWE-347": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Improper Signature Verification
    "CWE-523": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # Unprotected Credentials
    "CWE-720": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,  # OWASP Top Ten 2007 A8
    # A03: Injection
    "CWE-20": OWASPCategory.A03_INJECTION,  # Improper Input Validation
    "CWE-74": OWASPCategory.A03_INJECTION,  # Injection
    "CWE-75": OWASPCategory.A03_INJECTION,  # Failure to Sanitize Special Elements
    "CWE-77": OWASPCategory.A03_INJECTION,  # Command Injection
    "CWE-78": OWASPCategory.A03_INJECTION,  # OS Command Injection
    "CWE-79": OWASPCategory.A03_INJECTION,  # XSS
    "CWE-80": OWASPCategory.A03_INJECTION,  # Basic XSS
    "CWE-83": OWASPCategory.A03_INJECTION,  # XSS in Script Block
    "CWE-87": OWASPCategory.A03_INJECTION,  # XSS in Alternate Syntax
    "CWE-88": OWASPCategory.A03_INJECTION,  # Argument Injection
    "CWE-89": OWASPCategory.A03_INJECTION,  # SQL Injection
    "CWE-90": OWASPCategory.A03_INJECTION,  # LDAP Injection
    "CWE-91": OWASPCategory.A03_INJECTION,  # XML Injection
    "CWE-93": OWASPCategory.A03_INJECTION,  # CRLF Injection
    "CWE-94": OWASPCategory.A03_INJECTION,  # Code Injection
    "CWE-95": OWASPCategory.A03_INJECTION,  # Eval Injection
    "CWE-96": OWASPCategory.A03_INJECTION,  # Static Code Injection
    "CWE-97": OWASPCategory.A03_INJECTION,  # Server-Side Include Injection
    "CWE-98": OWASPCategory.A03_INJECTION,  # PHP Remote File Inclusion
    "CWE-99": OWASPCategory.A03_INJECTION,  # Resource Injection
    "CWE-113": OWASPCategory.A03_INJECTION,  # HTTP Response Splitting
    "CWE-116": OWASPCategory.A03_INJECTION,  # Improper Encoding
    "CWE-138": OWASPCategory.A03_INJECTION,  # Improper Neutralization of Special Elements
    "CWE-564": OWASPCategory.A03_INJECTION,  # SQL Injection: Hibernate
    "CWE-611": OWASPCategory.A03_INJECTION,  # XXE
    "CWE-643": OWASPCategory.A03_INJECTION,  # XPath Injection
    "CWE-652": OWASPCategory.A03_INJECTION,  # XQuery Injection
    "CWE-917": OWASPCategory.A03_INJECTION,  # Expression Language Injection
    # A04: Insecure Design
    "CWE-73": OWASPCategory.A04_INSECURE_DESIGN,  # External Control of File Name
    "CWE-183": OWASPCategory.A04_INSECURE_DESIGN,  # Permissive Whitelist
    "CWE-209": OWASPCategory.A04_INSECURE_DESIGN,  # Error Message Information Exposure
    "CWE-213": OWASPCategory.A04_INSECURE_DESIGN,  # Intentional Information Exposure
    "CWE-235": OWASPCategory.A04_INSECURE_DESIGN,  # Improper Handling of Extra Parameters
    "CWE-256": OWASPCategory.A04_INSECURE_DESIGN,  # Plaintext Storage of Password
    "CWE-257": OWASPCategory.A04_INSECURE_DESIGN,  # Storing Passwords in Recoverable Format
    "CWE-266": OWASPCategory.A04_INSECURE_DESIGN,  # Incorrect Privilege Assignment
    "CWE-269": OWASPCategory.A04_INSECURE_DESIGN,  # Improper Privilege Management
    "CWE-280": OWASPCategory.A04_INSECURE_DESIGN,  # Improper Handling of Insufficient Privileges
    "CWE-311": OWASPCategory.A04_INSECURE_DESIGN,  # Missing Encryption of Sensitive Data
    "CWE-312": OWASPCategory.A04_INSECURE_DESIGN,  # Cleartext Storage of Sensitive Info
    "CWE-313": OWASPCategory.A04_INSECURE_DESIGN,  # Cleartext Storage in File
    "CWE-316": OWASPCategory.A04_INSECURE_DESIGN,  # Cleartext Storage in Memory
    "CWE-419": OWASPCategory.A04_INSECURE_DESIGN,  # Unprotected Primary Channel
    "CWE-430": OWASPCategory.A04_INSECURE_DESIGN,  # Deployment of Wrong Handler
    "CWE-434": OWASPCategory.A04_INSECURE_DESIGN,  # Unrestricted Upload
    "CWE-444": OWASPCategory.A04_INSECURE_DESIGN,  # HTTP Request Smuggling
    "CWE-451": OWASPCategory.A04_INSECURE_DESIGN,  # UI Misrepresentation
    "CWE-472": OWASPCategory.A04_INSECURE_DESIGN,  # External Control of Assumed-Immutable
    "CWE-501": OWASPCategory.A04_INSECURE_DESIGN,  # Trust Boundary Violation
    "CWE-522": OWASPCategory.A04_INSECURE_DESIGN,  # Insufficiently Protected Credentials
    "CWE-525": OWASPCategory.A04_INSECURE_DESIGN,  # Information Exposure Through Browser Caching
    "CWE-539": OWASPCategory.A04_INSECURE_DESIGN,  # Information Exposure Through Persistent Cookies
    "CWE-579": OWASPCategory.A04_INSECURE_DESIGN,  # J2EE Bad Practices
    "CWE-598": OWASPCategory.A04_INSECURE_DESIGN,  # Information Exposure Through Query Strings
    "CWE-602": OWASPCategory.A04_INSECURE_DESIGN,  # Client-Side Enforcement of Server-Side Security
    "CWE-642": OWASPCategory.A04_INSECURE_DESIGN,  # External Control of Critical State Data
    "CWE-646": OWASPCategory.A04_INSECURE_DESIGN,  # Reliance on File Name or Extension
    "CWE-650": OWASPCategory.A04_INSECURE_DESIGN,  # Trusting HTTP Permission Methods
    "CWE-653": OWASPCategory.A04_INSECURE_DESIGN,  # Insufficient Compartmentalization
    "CWE-656": OWASPCategory.A04_INSECURE_DESIGN,  # Reliance on Security Through Obscurity
    "CWE-657": OWASPCategory.A04_INSECURE_DESIGN,  # Violation of Secure Design Principles
    "CWE-799": OWASPCategory.A04_INSECURE_DESIGN,  # Improper Control of Interaction Frequency
    "CWE-807": OWASPCategory.A04_INSECURE_DESIGN,  # Reliance on Untrusted Inputs
    "CWE-840": OWASPCategory.A04_INSECURE_DESIGN,  # Business Logic Errors
    "CWE-841": OWASPCategory.A04_INSECURE_DESIGN,  # Improper Enforcement of Behavioral Workflow
    "CWE-927": OWASPCategory.A04_INSECURE_DESIGN,  # Implicit Intent
    # A05: Security Misconfiguration
    "CWE-2": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Environmental Security Flaw
    "CWE-11": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # ASP.NET Misconfiguration
    "CWE-13": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # ASP.NET Misconfiguration
    "CWE-15": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # External Control of System Setting
    "CWE-16": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Configuration
    "CWE-260": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Password in Configuration File
    "CWE-315": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Cleartext Storage in Cookie
    "CWE-520": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # .NET Misconfiguration
    "CWE-526": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Information Exposure Through Environment Variables
    "CWE-537": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Runtime Error Message Containing Sensitive Info
    "CWE-541": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Information Exposure Through Include Source Code
    "CWE-547": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Use of Hard-coded Security-relevant Constants
    # Note: CWE-611 (XXE) is mapped to A03_INJECTION above
    "CWE-614": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Sensitive Cookie Without Secure Attribute
    "CWE-756": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Missing Custom Error Page
    "CWE-776": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Recursive Entity Reference
    "CWE-942": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Permissive Cross-domain Policy
    "CWE-1004": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # Sensitive Cookie Without HttpOnly
    "CWE-1032": OWASPCategory.A05_SECURITY_MISCONFIGURATION,  # OWASP Top Ten 2017 A6
    # A07: Identification and Authentication Failures
    "CWE-255": OWASPCategory.A07_AUTH_FAILURES,  # Credentials Management
    "CWE-259": OWASPCategory.A07_AUTH_FAILURES,  # Hard-coded Password
    "CWE-287": OWASPCategory.A07_AUTH_FAILURES,  # Improper Authentication
    "CWE-288": OWASPCategory.A07_AUTH_FAILURES,  # Authentication Bypass Using Alternate Path
    "CWE-290": OWASPCategory.A07_AUTH_FAILURES,  # Authentication Bypass by Spoofing
    "CWE-294": OWASPCategory.A07_AUTH_FAILURES,  # Authentication Bypass by Capture-replay
    "CWE-295": OWASPCategory.A07_AUTH_FAILURES,  # Improper Certificate Validation
    "CWE-297": OWASPCategory.A07_AUTH_FAILURES,  # Improper Validation of Certificate with Host Mismatch
    "CWE-300": OWASPCategory.A07_AUTH_FAILURES,  # Channel Accessible by Non-Endpoint
    "CWE-302": OWASPCategory.A07_AUTH_FAILURES,  # Authentication Bypass by Assumed-Immutable Data
    "CWE-304": OWASPCategory.A07_AUTH_FAILURES,  # Missing Critical Step in Authentication
    "CWE-306": OWASPCategory.A07_AUTH_FAILURES,  # Missing Authentication for Critical Function
    "CWE-307": OWASPCategory.A07_AUTH_FAILURES,  # Improper Restriction of Excessive Authentication Attempts
    "CWE-346": OWASPCategory.A07_AUTH_FAILURES,  # Origin Validation Error
    "CWE-384": OWASPCategory.A07_AUTH_FAILURES,  # Session Fixation
    "CWE-521": OWASPCategory.A07_AUTH_FAILURES,  # Weak Password Requirements
    "CWE-613": OWASPCategory.A07_AUTH_FAILURES,  # Insufficient Session Expiration
    "CWE-620": OWASPCategory.A07_AUTH_FAILURES,  # Unverified Password Change
    "CWE-640": OWASPCategory.A07_AUTH_FAILURES,  # Weak Password Recovery Mechanism
    "CWE-798": OWASPCategory.A07_AUTH_FAILURES,  # Hard-coded Credentials
    # A08: Software and Data Integrity Failures
    "CWE-345": OWASPCategory.A08_DATA_INTEGRITY,  # Insufficient Verification of Data Authenticity
    "CWE-353": OWASPCategory.A08_DATA_INTEGRITY,  # Missing Support for Integrity Check
    "CWE-426": OWASPCategory.A08_DATA_INTEGRITY,  # Untrusted Search Path
    "CWE-494": OWASPCategory.A08_DATA_INTEGRITY,  # Download of Code Without Integrity Check
    "CWE-502": OWASPCategory.A08_DATA_INTEGRITY,  # Deserialization of Untrusted Data
    "CWE-565": OWASPCategory.A08_DATA_INTEGRITY,  # Reliance on Cookies without Validation and Integrity Checking
    "CWE-784": OWASPCategory.A08_DATA_INTEGRITY,  # Reliance on Cookies in Security Decision
    "CWE-829": OWASPCategory.A08_DATA_INTEGRITY,  # Inclusion of Functionality from Untrusted Control Sphere
    "CWE-830": OWASPCategory.A08_DATA_INTEGRITY,  # Inclusion of Web Functionality from Untrusted Source
    "CWE-915": OWASPCategory.A08_DATA_INTEGRITY,  # Improperly Controlled Modification of Dynamically-Determined Object Attributes
    # A09: Security Logging and Monitoring Failures
    "CWE-117": OWASPCategory.A09_LOGGING_FAILURES,  # Improper Output Neutralization for Logs
    "CWE-223": OWASPCategory.A09_LOGGING_FAILURES,  # Omission of Security-relevant Information
    "CWE-532": OWASPCategory.A09_LOGGING_FAILURES,  # Information Exposure Through Log Files
    "CWE-778": OWASPCategory.A09_LOGGING_FAILURES,  # Insufficient Logging
    # A10: Server-Side Request Forgery
    "CWE-918": OWASPCategory.A10_SSRF,  # Server-Side Request Forgery
}


# Local fallback patterns when Semgrep is not available
LOCAL_PATTERNS: Dict[str, Dict[str, Any]] = {
    # SQL Injection patterns
    "sql-injection-python": {
        "pattern": r'execute\s*\(\s*[\'"].*%s.*[\'"]\s*%',
        "languages": ["python"],
        "message": "Potential SQL injection via string formatting",
        "severity": SASTSeverity.CRITICAL,
        "cwe": "CWE-89",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    "sql-injection-fstring": {
        "pattern": r'execute\s*\(\s*f[\'"].*\{.*\}.*[\'"]',
        "languages": ["python"],
        "message": "Potential SQL injection via f-string",
        "severity": SASTSeverity.CRITICAL,
        "cwe": "CWE-89",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    # Command Injection
    "command-injection-subprocess": {
        "pattern": r'subprocess\.(call|run|Popen)\s*\(\s*[\'"].*\+.*shell\s*=\s*True',
        "languages": ["python"],
        "message": "Potential command injection with shell=True",
        "severity": SASTSeverity.CRITICAL,
        "cwe": "CWE-78",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    "command-injection-os-system": {
        "pattern": r"os\.system\s*\([^)]*\+[^)]*\)",
        "languages": ["python"],
        "message": "Potential command injection via os.system",
        "severity": SASTSeverity.CRITICAL,
        "cwe": "CWE-78",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    # Eval/Exec injection
    "eval-injection": {
        "pattern": r"\beval\s*\([^)]*\)",
        "languages": ["python", "javascript"],
        "message": "Use of eval() is dangerous and may allow code injection",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-95",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    "exec-injection": {
        "pattern": r"\bexec\s*\([^)]*\)",
        "languages": ["python"],
        "message": "Use of exec() is dangerous and may allow code injection",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-95",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    # Hardcoded credentials
    "hardcoded-password": {
        "pattern": r'(password|passwd|pwd)\s*=\s*[\'"][^\'"]{4,}[\'"]',
        "languages": ["python", "javascript", "java", "go"],
        "message": "Potential hardcoded password",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-259",
        "owasp": OWASPCategory.A07_AUTH_FAILURES,
    },
    "hardcoded-api-key": {
        "pattern": r'(api[_-]?key|apikey|api[_-]?secret)\s*=\s*[\'"][A-Za-z0-9_\-]{20,}[\'"]',
        "languages": ["python", "javascript", "java", "go"],
        "message": "Potential hardcoded API key",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-798",
        "owasp": OWASPCategory.A07_AUTH_FAILURES,
    },
    # XSS patterns
    "xss-innerHTML": {
        "pattern": r"\.innerHTML\s*=",
        "languages": ["javascript", "typescript"],
        "message": "Use of innerHTML may lead to XSS",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-79",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    "xss-document-write": {
        "pattern": r"document\.write\s*\(",
        "languages": ["javascript", "typescript"],
        "message": "Use of document.write may lead to XSS",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-79",
        "owasp": OWASPCategory.A03_INJECTION,
    },
    # Insecure deserialization
    "pickle-load": {
        "pattern": r"pickle\.loads?\s*\(",
        "languages": ["python"],
        "message": "Pickle deserialization of untrusted data is dangerous",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-502",
        "owasp": OWASPCategory.A08_DATA_INTEGRITY,
    },
    "yaml-unsafe-load": {
        "pattern": r"yaml\.load\s*\([^)]*\)(?!\s*,\s*Loader)",
        "languages": ["python"],
        "message": "YAML load without safe loader is dangerous",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-502",
        "owasp": OWASPCategory.A08_DATA_INTEGRITY,
    },
    # Weak cryptography
    "weak-hash-md5": {
        "pattern": r"(md5|MD5)\s*\(",
        "languages": ["python", "javascript", "java", "go"],
        "message": "MD5 is a weak hash algorithm",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-327",
        "owasp": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
    },
    "weak-hash-sha1": {
        "pattern": r"(sha1|SHA1)\s*\(",
        "languages": ["python", "javascript", "java", "go"],
        "message": "SHA1 is a weak hash algorithm",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-327",
        "owasp": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
    },
    # SSRF
    "ssrf-requests": {
        "pattern": r"requests\.(get|post|put|delete|head|options)\s*\([^)]*\+[^)]*\)",
        "languages": ["python"],
        "message": "Potential SSRF via user-controlled URL",
        "severity": SASTSeverity.ERROR,
        "cwe": "CWE-918",
        "owasp": OWASPCategory.A10_SSRF,
    },
    # Path traversal
    "path-traversal": {
        "pattern": r"open\s*\([^)]*\+[^)]*\)",
        "languages": ["python"],
        "message": "Potential path traversal via string concatenation in open()",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-22",
        "owasp": OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
    },
    # Debug enabled
    "debug-enabled": {
        "pattern": r"DEBUG\s*=\s*True",
        "languages": ["python"],
        "message": "Debug mode enabled in production may expose sensitive information",
        "severity": SASTSeverity.WARNING,
        "cwe": "CWE-215",
        "owasp": OWASPCategory.A05_SECURITY_MISCONFIGURATION,
    },
    # JWT without verification
    "jwt-no-verify": {
        "pattern": r"jwt\.decode\s*\([^)]*verify\s*=\s*False",
        "languages": ["python"],
        "message": "JWT decoded without verification",
        "severity": SASTSeverity.CRITICAL,
        "cwe": "CWE-347",
        "owasp": OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
    },
}

# Language extension mapping
LANGUAGE_EXTENSIONS: Dict[str, List[str]] = {
    "python": [".py", ".pyw"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "go": [".go"],
    "java": [".java"],
    "ruby": [".rb"],
    "php": [".php"],
    "csharp": [".cs"],
}


class SASTScanner:
    """
    Static Application Security Testing scanner.

    Integrates with Semgrep for comprehensive static analysis.
    Falls back to local pattern matching when Semgrep is unavailable.

    Features:
    - OWASP Top 10 rule pack support
    - CWE ID mapping for all findings
    - Multi-language support (Python, JavaScript, Go, Java, TypeScript)
    - False positive filtering via confidence scores
    - Async scanning with progress reporting
    - SecurityEventEmitter integration for critical findings
    """

    # Semgrep installation instructions
    SEMGREP_INSTALL_INSTRUCTIONS = """
Semgrep is not installed or not available in PATH.

To install Semgrep, use one of the following methods:

1. Using pip (recommended):
   pip install semgrep

2. Using Homebrew (macOS):
   brew install semgrep

3. Using Docker:
   docker pull returntocorp/semgrep

4. Using pipx (isolated installation):
   pipx install semgrep

For more information, visit: https://semgrep.dev/docs/getting-started/

The scanner will fall back to local pattern matching until Semgrep is installed.
"""

    def __init__(
        self,
        config: Optional[SASTConfig] = None,
        security_emitter: Optional[Any] = None,
    ):
        """
        Initialize SAST scanner.

        Args:
            config: Scanner configuration
            security_emitter: Optional SecurityEventEmitter for critical finding notifications
        """
        self.config = config or SASTConfig()
        self._semgrep_available: Optional[bool] = None
        self._semgrep_version: Optional[str] = None
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._security_emitter = security_emitter
        self._scan_progress: Dict[str, int] = {}

        # Compile local patterns
        for rule_id, rule_data in LOCAL_PATTERNS.items():
            try:
                self._compiled_patterns[rule_id] = re.compile(
                    rule_data["pattern"],
                    re.IGNORECASE | re.MULTILINE,
                )
            except re.error as e:
                logger.warning(f"Failed to compile pattern {rule_id}: {e}")

    async def initialize(self) -> None:
        """Initialize scanner and check Semgrep availability."""
        if self.config.use_semgrep:
            self._semgrep_available, self._semgrep_version = await self._check_semgrep()
            if self._semgrep_available:
                logger.info(f"Semgrep {self._semgrep_version} is available")
            else:
                logger.warning("Semgrep not available, using local patterns")
                logger.info(self.SEMGREP_INSTALL_INSTRUCTIONS)
        else:
            self._semgrep_available = False

        # Initialize security emitter if configured
        if self.config.emit_security_events and self._security_emitter is None:
            try:
                from aragora.events.security_events import get_security_emitter

                self._security_emitter = get_security_emitter()
            except ImportError:
                logger.debug("SecurityEventEmitter not available")

    async def _check_semgrep(self) -> tuple[bool, Optional[str]]:
        """Check if Semgrep is installed and accessible."""
        try:
            process = await asyncio.create_subprocess_exec(
                self.config.semgrep_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)
            if process.returncode == 0:
                version = stdout.decode().strip().split("\n")[0]
                return True, version
            return False, None
        except Exception as e:
            logger.debug(f"Semgrep check failed: {e}")
            return False, None

    def is_semgrep_available(self) -> bool:
        """Check if Semgrep is available for scanning."""
        return self._semgrep_available or False

    def get_semgrep_version(self) -> Optional[str]:
        """Get the installed Semgrep version."""
        return self._semgrep_version

    def get_install_instructions(self) -> str:
        """Get Semgrep installation instructions."""
        return self.SEMGREP_INSTALL_INSTRUCTIONS

    async def get_available_rulesets(self) -> List[Dict[str, Any]]:
        """
        Get available Semgrep rulesets.

        Returns:
            List of available rulesets with name, description, and category
        """
        rulesets = []

        for ruleset_id, ruleset_info in AVAILABLE_RULESETS.items():
            rulesets.append(
                {
                    "id": ruleset_id,
                    "name": ruleset_info["name"],
                    "description": ruleset_info["description"],
                    "category": ruleset_info["category"],
                    "available": self._semgrep_available or False,
                }
            )

        # If Semgrep is available, try to get additional rulesets from registry
        if self._semgrep_available:
            try:
                additional = await self._fetch_registry_rulesets()
                # Merge with existing, avoiding duplicates
                existing_ids = {r["id"] for r in rulesets}
                for ruleset in additional:
                    if ruleset["id"] not in existing_ids:
                        rulesets.append(ruleset)
            except Exception as e:
                logger.debug(f"Failed to fetch registry rulesets: {e}")

        return rulesets

    async def _fetch_registry_rulesets(self) -> List[Dict[str, Any]]:
        """Fetch available rulesets from Semgrep registry."""
        # This is a simplified implementation
        # In production, you might want to cache this and refresh periodically
        return []  # Registry fetch would go here

    async def scan_repository(
        self,
        repo_path: str,
        rule_sets: Optional[List[str]] = None,
        scan_id: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        min_confidence: Optional[float] = None,
    ) -> SASTScanResult:
        """
        Scan a repository for security issues.

        Args:
            repo_path: Path to repository
            rule_sets: Optional list of rule sets to use
            scan_id: Optional scan identifier
            progress_callback: Optional async callback for progress updates
            min_confidence: Minimum confidence threshold for findings (0.0-1.0)

        Returns:
            SASTScanResult with findings
        """
        start_time = datetime.now()
        scan_id = scan_id or str(uuid.uuid4())[:8]
        repo_path = os.path.abspath(repo_path)

        if not os.path.isdir(repo_path):
            return SASTScanResult(
                repository_path=repo_path,
                scan_id=scan_id,
                findings=[],
                scanned_files=0,
                skipped_files=0,
                scan_duration_ms=0,
                languages_detected=[],
                rules_used=[],
                errors=[f"Repository path not found: {repo_path}"],
            )

        rule_sets = rule_sets or self.config.default_rule_sets
        confidence_threshold = min_confidence or self.config.min_confidence_threshold

        # Report initial progress
        if progress_callback:
            await progress_callback(0, 100, f"Starting scan of {os.path.basename(repo_path)}")

        # Try Semgrep first, fall back to local patterns
        if self._semgrep_available:
            if progress_callback:
                await progress_callback(10, 100, "Running Semgrep analysis...")
            result = await self._scan_with_semgrep(repo_path, rule_sets, scan_id)
        else:
            if progress_callback:
                await progress_callback(10, 100, "Running local pattern analysis...")
            result = await self._scan_with_local_patterns(repo_path, scan_id, progress_callback)

        # Apply false positive filtering
        if self.config.enable_false_positive_filtering:
            original_count = len(result.findings)
            result.findings = [f for f in result.findings if f.confidence >= confidence_threshold]
            filtered_count = original_count - len(result.findings)
            if filtered_count > 0:
                logger.debug(
                    f"Filtered {filtered_count} low-confidence findings "
                    f"(threshold: {confidence_threshold})"
                )

        # Add fix recommendations based on CWE
        for finding in result.findings:
            if not finding.remediation and finding.cwe_ids:
                for cwe_id in finding.cwe_ids:
                    if cwe_id in CWE_FIX_RECOMMENDATIONS:
                        finding.remediation = CWE_FIX_RECOMMENDATIONS[cwe_id]
                        break

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds() * 1000
        result.scan_duration_ms = duration

        # Report completion
        if progress_callback:
            await progress_callback(100, 100, f"Scan complete: {len(result.findings)} findings")

        # Emit security events for critical findings
        await self._emit_security_events(result, repo_path, scan_id)

        return result

    async def _emit_security_events(
        self,
        result: SASTScanResult,
        repo_path: str,
        scan_id: str,
    ) -> None:
        """Emit security events for critical findings."""
        if not self.config.emit_security_events or not self._security_emitter:
            return

        # Count critical and high severity findings
        critical_findings = [f for f in result.findings if f.severity == SASTSeverity.CRITICAL]
        _high_findings = [f for f in result.findings if f.severity == SASTSeverity.ERROR]

        # Emit event if threshold is met
        if len(critical_findings) >= self.config.critical_finding_threshold:
            try:
                from aragora.events.security_events import (
                    SecurityEvent,
                    SecurityEventType,
                    SecuritySeverity,
                    SecurityFinding,
                )

                # Convert SAST findings to security findings
                security_findings = []
                for f in critical_findings[:10]:  # Limit to top 10
                    security_findings.append(
                        SecurityFinding(
                            id=f.finding_id,
                            finding_type="vulnerability",
                            severity=SecuritySeverity.CRITICAL,
                            title=f.rule_name or f.rule_id,
                            description=f.message,
                            file_path=f.file_path,
                            line_number=f.line_start,
                            recommendation=f.remediation,
                            metadata={
                                "cwe_ids": f.cwe_ids,
                                "owasp_category": f.owasp_category.value,
                                "snippet": f.snippet[:200] if f.snippet else "",
                                "source": f.source,
                            },
                        )
                    )

                event = SecurityEvent(
                    event_type=SecurityEventType.CRITICAL_VULNERABILITY,
                    severity=SecuritySeverity.CRITICAL,
                    repository=os.path.basename(repo_path),
                    scan_id=scan_id,
                    findings=security_findings,
                )

                await self._security_emitter.emit(event)
                logger.info(
                    f"Emitted security event for {len(critical_findings)} critical findings"
                )

            except ImportError:
                logger.debug("SecurityEventEmitter not available for event emission")
            except Exception as e:
                logger.warning(f"Failed to emit security event: {e}")

    async def _scan_with_semgrep(
        self,
        repo_path: str,
        rule_sets: List[str],
        scan_id: str,
    ) -> SASTScanResult:
        """Run Semgrep scan on repository."""
        findings: List[SASTFinding] = []
        errors: List[str] = []
        languages_detected: Set[str] = set()

        try:
            # Build Semgrep command
            cmd = [
                self.config.semgrep_path,
                "--json",
                "--metrics=off",
                "--timeout",
                str(self.config.semgrep_timeout),
            ]

            # Add rule sets
            for rule_set in rule_sets:
                cmd.extend(["--config", rule_set])

            # Add custom rules if configured
            if self.config.custom_rules_dir and os.path.isdir(self.config.custom_rules_dir):
                cmd.extend(["--config", self.config.custom_rules_dir])

            # Add exclusions
            for pattern in self.config.excluded_patterns:
                cmd.extend(["--exclude", pattern])

            # Add target
            cmd.append(repo_path)

            logger.info(f"Running Semgrep scan: {' '.join(cmd[:5])}...")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.semgrep_timeout + 30,
            )

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if "error" in stderr_text.lower():
                    errors.append(stderr_text[:500])

            if stdout:
                output = json.loads(stdout.decode("utf-8"))

                # Parse results
                for result in output.get("results", []):
                    finding = self._parse_semgrep_result(result)
                    if finding and finding.severity.value >= self.config.min_severity.value:
                        findings.append(finding)
                        languages_detected.add(finding.language)

                # Get scan stats
                paths = output.get("paths", {})
                scanned_files = len(paths.get("scanned", []))
                skipped_files = len(paths.get("skipped", []))

                return SASTScanResult(
                    repository_path=repo_path,
                    scan_id=scan_id,
                    findings=findings,
                    scanned_files=scanned_files,
                    skipped_files=skipped_files,
                    scan_duration_ms=0,
                    languages_detected=list(languages_detected),
                    rules_used=rule_sets,
                    errors=errors,
                )

        except asyncio.TimeoutError:
            errors.append("Semgrep scan timed out")
        except json.JSONDecodeError as e:
            errors.append(f"Failed to parse Semgrep output: {e}")
        except Exception as e:
            errors.append(f"Semgrep scan failed: {e}")

        # Return partial result on error
        return SASTScanResult(
            repository_path=repo_path,
            scan_id=scan_id,
            findings=findings,
            scanned_files=0,
            skipped_files=0,
            scan_duration_ms=0,
            languages_detected=list(languages_detected),
            rules_used=rule_sets,
            errors=errors,
        )

    def _parse_semgrep_result(self, result: Dict[str, Any]) -> Optional[SASTFinding]:
        """Parse a single Semgrep result into a SASTFinding."""
        try:
            check_id = result.get("check_id", "unknown")
            path = result.get("path", "")
            start = result.get("start", {})
            end = result.get("end", {})
            extra = result.get("extra", {})
            metadata = extra.get("metadata", {})

            # Get severity
            severity_str = extra.get("severity", "WARNING").upper()
            severity = getattr(SASTSeverity, severity_str, SASTSeverity.WARNING)

            # Get CWE IDs
            cwe_ids = metadata.get("cwe", [])
            if isinstance(cwe_ids, str):
                cwe_ids = [cwe_ids]

            # Map to OWASP
            owasp_category = OWASPCategory.UNKNOWN
            for cwe in cwe_ids:
                if cwe in CWE_TO_OWASP:
                    owasp_category = CWE_TO_OWASP[cwe]
                    break

            # Also check OWASP directly from metadata
            owasp_str = metadata.get("owasp", "")
            if owasp_str and owasp_category == OWASPCategory.UNKNOWN:
                for cat in OWASPCategory:
                    if cat.value.startswith(owasp_str[:3]):
                        owasp_category = cat
                        break

            return SASTFinding(
                rule_id=check_id,
                file_path=path,
                line_start=start.get("line", 0),
                line_end=end.get("line", 0),
                column_start=start.get("col", 0),
                column_end=end.get("col", 0),
                message=extra.get("message", ""),
                severity=severity,
                confidence=metadata.get("confidence", 0.8),
                language=metadata.get("language", self._detect_language(path)),
                snippet=extra.get("lines", ""),
                cwe_ids=cwe_ids,
                owasp_category=owasp_category,
                vulnerability_class=metadata.get("vulnerability_class", ""),
                remediation=metadata.get("fix", ""),
                source="semgrep",
                rule_name=check_id.split(".")[-1],
                rule_url=metadata.get("source", ""),
                metadata=metadata,
            )

        except Exception as e:
            logger.warning(f"Failed to parse Semgrep result: {e}")
            return None

    async def _scan_with_local_patterns(
        self,
        repo_path: str,
        scan_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SASTScanResult:
        """Scan repository using local patterns (fallback)."""
        findings: List[SASTFinding] = []
        languages_detected: Set[str] = set()
        scanned_files = 0
        skipped_files = 0
        errors: List[str] = []

        # First pass: count files for progress reporting
        files_to_scan: List[tuple[str, str, str]] = []

        try:
            # Walk the repository
            for root, dirs, files in os.walk(repo_path):
                # Skip excluded directories
                dirs[:] = [
                    d
                    for d in dirs
                    if not any(
                        d == p.rstrip("/") or f"{d}/" == p for p in self.config.excluded_patterns
                    )
                ]

                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, repo_path)

                    # Skip excluded files
                    if any(
                        re.match(p.replace("*", ".*"), rel_path)
                        for p in self.config.excluded_patterns
                    ):
                        skipped_files += 1
                        continue

                    # Detect language
                    language = self._detect_language(filename)
                    if language not in self.config.supported_languages:
                        skipped_files += 1
                        continue

                    # Check file size
                    try:
                        size_kb = os.path.getsize(file_path) / 1024
                        if size_kb > self.config.max_file_size_kb:
                            skipped_files += 1
                            continue
                    except OSError:
                        continue

                    files_to_scan.append((file_path, rel_path, language))

            total_files = len(files_to_scan)
            if progress_callback and total_files > 0:
                await progress_callback(15, 100, f"Scanning {total_files} files...")

            # Second pass: scan files with progress
            for idx, (file_path, rel_path, language) in enumerate(files_to_scan):
                file_findings = await self._scan_file_local(file_path, rel_path, language)
                findings.extend(file_findings)
                scanned_files += 1

                if file_findings:
                    languages_detected.add(language)

                # Report progress periodically
                if progress_callback and total_files > 0 and idx % 10 == 0:
                    progress = 15 + int((idx / total_files) * 80)
                    await progress_callback(progress, 100, f"Scanned {idx + 1}/{total_files} files")

        except Exception as e:
            errors.append(f"Local scan error: {e}")

        return SASTScanResult(
            repository_path=repo_path,
            scan_id=scan_id,
            findings=findings,
            scanned_files=scanned_files,
            skipped_files=skipped_files,
            scan_duration_ms=0,
            languages_detected=list(languages_detected),
            rules_used=list(LOCAL_PATTERNS.keys()),
            errors=errors,
        )

    async def _scan_file_local(
        self,
        file_path: str,
        rel_path: str,
        language: str,
    ) -> List[SASTFinding]:
        """Scan a single file with local patterns."""
        findings: List[SASTFinding] = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

            for rule_id, pattern in self._compiled_patterns.items():
                rule_data = LOCAL_PATTERNS[rule_id]

                # Check if language matches
                if language not in rule_data.get("languages", []):
                    continue

                # Find matches
                for match in pattern.finditer(content):
                    start_pos = match.start()
                    end_pos = match.end()

                    # Calculate line numbers
                    line_start = content[:start_pos].count("\n") + 1
                    line_end = content[:end_pos].count("\n") + 1

                    # Get snippet (context lines)
                    snippet_start = max(0, line_start - 2)
                    snippet_end = min(len(lines), line_end + 2)
                    snippet = "\n".join(lines[snippet_start:snippet_end])

                    finding = SASTFinding(
                        rule_id=rule_id,
                        file_path=rel_path,
                        line_start=line_start,
                        line_end=line_end,
                        column_start=match.start() - content.rfind("\n", 0, start_pos) - 1,
                        column_end=match.end() - content.rfind("\n", 0, end_pos) - 1,
                        message=rule_data["message"],
                        severity=rule_data["severity"],
                        confidence=0.7,  # Lower confidence for local patterns
                        language=language,
                        snippet=snippet,
                        cwe_ids=[rule_data["cwe"]],
                        owasp_category=rule_data["owasp"],
                        source="local",
                        rule_name=rule_id,
                    )

                    # Filter by severity
                    if finding.severity >= self.config.min_severity:
                        findings.append(finding)

        except Exception as e:
            logger.debug(f"Error scanning {file_path}: {e}")

        return findings

    def _detect_language(self, filename: str) -> str:
        """Detect programming language from filename."""
        ext = os.path.splitext(filename)[1].lower()
        for lang, extensions in LANGUAGE_EXTENSIONS.items():
            if ext in extensions:
                return lang
        return "unknown"

    async def scan_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[SASTFinding]:
        """
        Scan a single file for security issues.

        Args:
            file_path: Path to file
            language: Optional language override
            min_confidence: Minimum confidence threshold for findings

        Returns:
            List of findings
        """
        if not os.path.isfile(file_path):
            return []

        language = language or self._detect_language(file_path)
        rel_path = os.path.basename(file_path)
        confidence_threshold = min_confidence or self.config.min_confidence_threshold

        if self._semgrep_available:
            # Use Semgrep for single file
            result = await self._scan_with_semgrep(
                os.path.dirname(file_path),
                self.config.default_rule_sets,
                "single",
            )
            findings = [f for f in result.findings if f.file_path.endswith(rel_path)]
        else:
            findings = await self._scan_file_local(file_path, rel_path, language)

        # Apply confidence filtering
        if self.config.enable_false_positive_filtering:
            findings = [f for f in findings if f.confidence >= confidence_threshold]

        # Add fix recommendations
        for finding in findings:
            if not finding.remediation and finding.cwe_ids:
                for cwe_id in finding.cwe_ids:
                    if cwe_id in CWE_FIX_RECOMMENDATIONS:
                        finding.remediation = CWE_FIX_RECOMMENDATIONS[cwe_id]
                        break

        return findings

    async def get_owasp_summary(
        self,
        findings: List[SASTFinding],
    ) -> Dict[str, Any]:
        """
        Generate OWASP Top 10 summary from findings.

        Args:
            findings: List of SAST findings

        Returns:
            Summary organized by OWASP category
        """
        summary: Dict[str, Dict[str, Any]] = {}

        for cat in OWASPCategory:
            if cat == OWASPCategory.UNKNOWN:
                continue
            summary[cat.value] = {
                "count": 0,
                "critical": 0,
                "error": 0,
                "warning": 0,
                "findings": [],
            }

        for finding in findings:
            cat_key = finding.owasp_category.value
            if cat_key in summary:
                summary[cat_key]["count"] += 1
                summary[cat_key][finding.severity.value] = (
                    summary[cat_key].get(finding.severity.value, 0) + 1
                )
                if len(summary[cat_key]["findings"]) < 5:  # Top 5 examples
                    summary[cat_key]["findings"].append(
                        {
                            "file": finding.file_path,
                            "line": finding.line_start,
                            "message": finding.message[:100],
                        }
                    )

        # Sort by count
        sorted_summary = dict(sorted(summary.items(), key=lambda x: x[1]["count"], reverse=True))

        return {
            "owasp_top_10": sorted_summary,
            "total_findings": len(findings),
            "most_common": list(sorted_summary.keys())[:3],
        }


# Convenience function for quick scans
async def scan_for_vulnerabilities(
    path: str,
    rule_sets: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    progress_callback: Optional[ProgressCallback] = None,
) -> SASTScanResult:
    """
    Quick convenience function for SAST scanning.

    Args:
        path: Path to file or directory
        rule_sets: Optional rule sets
        min_confidence: Minimum confidence threshold for findings
        progress_callback: Optional async callback for progress updates

    Returns:
        SASTScanResult
    """
    scanner = SASTScanner()
    await scanner.initialize()

    if os.path.isfile(path):
        findings = await scanner.scan_file(path, min_confidence=min_confidence)
        return SASTScanResult(
            repository_path=path,
            scan_id="quick",
            findings=findings,
            scanned_files=1,
            skipped_files=0,
            scan_duration_ms=0,
            languages_detected=[scanner._detect_language(path)],
            rules_used=rule_sets or ["local"],
        )
    else:
        return await scanner.scan_repository(
            path,
            rule_sets,
            progress_callback=progress_callback,
            min_confidence=min_confidence,
        )


async def get_available_rulesets() -> List[Dict[str, Any]]:
    """
    Get available Semgrep rulesets.

    Returns:
        List of available rulesets with metadata
    """
    scanner = SASTScanner()
    await scanner.initialize()
    return await scanner.get_available_rulesets()


def check_semgrep_installation() -> Dict[str, Any]:
    """
    Check Semgrep installation status synchronously.

    Returns:
        Dictionary with installation status and instructions
    """
    import subprocess

    try:
        result = subprocess.run(
            ["semgrep", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return {
                "installed": True,
                "version": result.stdout.strip().split("\n")[0],
                "message": "Semgrep is installed and available",
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return {
        "installed": False,
        "version": None,
        "message": "Semgrep is not installed",
        "instructions": SASTScanner.SEMGREP_INSTALL_INSTRUCTIONS,
    }


__all__ = [
    # Main classes
    "SASTScanner",
    "SASTScanResult",
    "SASTFinding",
    "SASTSeverity",
    "SASTConfig",
    "OWASPCategory",
    # Convenience functions
    "scan_for_vulnerabilities",
    "get_available_rulesets",
    "check_semgrep_installation",
    # Type aliases
    "ProgressCallback",
    # Constants
    "AVAILABLE_RULESETS",
    "CWE_TO_OWASP",
    "CWE_FIX_RECOMMENDATIONS",
]
