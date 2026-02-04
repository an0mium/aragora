"""
SAST Scanner models - data classes and enums.

Contains:
- SASTSeverity (Enum)
- OWASPCategory (Enum)
- SASTFinding (dataclass)
- SASTScanResult (dataclass)
- SASTConfig (dataclass)
- ProgressCallback type alias
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine

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
    cwe_ids: list[str] = field(default_factory=list)
    owasp_category: OWASPCategory = OWASPCategory.UNKNOWN
    vulnerability_class: str = ""
    remediation: str = ""

    # Source information
    source: str = "semgrep"  # semgrep, local, custom
    rule_name: str = ""
    rule_url: str = ""

    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)
    is_false_positive: bool = False
    triaged: bool = False

    # Finding ID for tracking
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def to_dict(self) -> dict[str, Any]:
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
    findings: list[SASTFinding]
    scanned_files: int
    skipped_files: int
    scan_duration_ms: float
    languages_detected: list[str]
    rules_used: list[str]
    errors: list[str] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=datetime.now)

    @property
    def findings_by_severity(self) -> dict[str, int]:
        """Count findings by severity."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            sev = finding.severity.value
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    @property
    def findings_by_owasp(self) -> dict[str, int]:
        """Count findings by OWASP category."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            cat = finding.owasp_category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
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
    default_rule_sets: list[str] = field(
        default_factory=lambda: [
            "p/owasp-top-ten",
            "p/security-audit",
        ]
    )

    # Custom rules directory
    custom_rules_dir: str | None = None

    # File filters
    max_file_size_kb: int = 500
    excluded_patterns: list[str] = field(
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
    supported_languages: list[str] = field(
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
