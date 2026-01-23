"""
Data models for codebase security analysis.

Defines dataclasses for vulnerability findings, dependencies, scan results,
and code quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class VulnerabilitySeverity(str, Enum):
    """Severity level of a vulnerability."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

    @classmethod
    def from_cvss(cls, score: float) -> "VulnerabilitySeverity":
        """Convert CVSS score to severity level."""
        if score >= 9.0:
            return cls.CRITICAL
        elif score >= 7.0:
            return cls.HIGH
        elif score >= 4.0:
            return cls.MEDIUM
        elif score > 0:
            return cls.LOW
        return cls.UNKNOWN


class VulnerabilitySource(str, Enum):
    """Source of vulnerability data."""

    NVD = "nvd"  # NIST National Vulnerability Database
    OSV = "osv"  # Open Source Vulnerabilities
    GITHUB = "github"  # GitHub Security Advisories
    SNYK = "snyk"
    CUSTOM = "custom"


@dataclass
class VulnerabilityReference:
    """A reference link for a vulnerability."""

    url: str
    source: str
    tags: List[str] = field(default_factory=list)


@dataclass
class VulnerabilityFinding:
    """
    A vulnerability finding from security analysis.

    Contains CVE data, affected packages, and remediation info.
    """

    id: str  # CVE ID or advisory ID
    title: str
    description: str
    severity: VulnerabilitySeverity
    cvss_score: Optional[float] = None
    cvss_vector: Optional[str] = None

    # Affected package info
    package_name: Optional[str] = None
    package_ecosystem: Optional[str] = None  # npm, pypi, maven, etc.
    vulnerable_versions: List[str] = field(default_factory=list)
    patched_versions: List[str] = field(default_factory=list)

    # Source and metadata
    source: VulnerabilitySource = VulnerabilitySource.NVD
    published_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    references: List[VulnerabilityReference] = field(default_factory=list)
    cwe_ids: List[str] = field(default_factory=list)

    # Location in codebase
    file_path: Optional[str] = None
    line_number: Optional[int] = None

    # Remediation
    fix_available: bool = False
    recommended_version: Optional[str] = None
    remediation_guidance: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "cvss_score": self.cvss_score,
            "cvss_vector": self.cvss_vector,
            "package_name": self.package_name,
            "package_ecosystem": self.package_ecosystem,
            "vulnerable_versions": self.vulnerable_versions,
            "patched_versions": self.patched_versions,
            "source": self.source.value,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "references": [
                {"url": r.url, "source": r.source, "tags": r.tags} for r in self.references
            ],
            "cwe_ids": self.cwe_ids,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "fix_available": self.fix_available,
            "recommended_version": self.recommended_version,
            "remediation_guidance": self.remediation_guidance,
        }


@dataclass
class DependencyInfo:
    """Information about a project dependency."""

    name: str
    version: str
    ecosystem: str  # npm, pypi, maven, cargo, go, etc.
    direct: bool = True  # False if transitive
    dev_dependency: bool = False
    license: Optional[str] = None
    vulnerabilities: List[VulnerabilityFinding] = field(default_factory=list)
    parent: Optional[str] = None  # Parent package for transitive deps
    file_path: Optional[str] = None  # package.json, requirements.txt, etc.

    @property
    def has_vulnerabilities(self) -> bool:
        return len(self.vulnerabilities) > 0

    @property
    def highest_severity(self) -> Optional[VulnerabilitySeverity]:
        if not self.vulnerabilities:
            return None
        severity_order = [
            VulnerabilitySeverity.CRITICAL,
            VulnerabilitySeverity.HIGH,
            VulnerabilitySeverity.MEDIUM,
            VulnerabilitySeverity.LOW,
        ]
        for severity in severity_order:
            if any(v.severity == severity for v in self.vulnerabilities):
                return severity
        return VulnerabilitySeverity.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "ecosystem": self.ecosystem,
            "direct": self.direct,
            "dev_dependency": self.dev_dependency,
            "license": self.license,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "parent": self.parent,
            "file_path": self.file_path,
            "has_vulnerabilities": self.has_vulnerabilities,
            "highest_severity": self.highest_severity.value if self.highest_severity else None,
        }


@dataclass
class ScanResult:
    """Result of a security scan."""

    scan_id: str
    repository: str
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    error: Optional[str] = None

    # Findings
    dependencies: List[DependencyInfo] = field(default_factory=list)
    vulnerabilities: List[VulnerabilityFinding] = field(default_factory=list)

    # Summary counts
    total_dependencies: int = 0
    vulnerable_dependencies: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    def calculate_summary(self) -> None:
        """Calculate summary statistics from findings."""
        self.total_dependencies = len(self.dependencies)
        self.vulnerable_dependencies = sum(1 for d in self.dependencies if d.has_vulnerabilities)

        self.critical_count = sum(
            1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL
        )
        self.high_count = sum(
            1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.HIGH
        )
        self.medium_count = sum(
            1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.MEDIUM
        )
        self.low_count = sum(
            1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.LOW
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scan_id": self.scan_id,
            "repository": self.repository,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "error": self.error,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "summary": {
                "total_dependencies": self.total_dependencies,
                "vulnerable_dependencies": self.vulnerable_dependencies,
                "critical_count": self.critical_count,
                "high_count": self.high_count,
                "medium_count": self.medium_count,
                "low_count": self.low_count,
            },
        }


class MetricType(str, Enum):
    """Type of code quality metric."""

    COMPLEXITY = "complexity"  # Cyclomatic complexity
    MAINTAINABILITY = "maintainability"
    TEST_COVERAGE = "test_coverage"
    DUPLICATION = "duplication"
    LINES_OF_CODE = "lines_of_code"
    DOCUMENTATION = "documentation"
    SECURITY = "security"


@dataclass
class CodeMetric:
    """A code quality metric."""

    type: MetricType
    value: float
    unit: str = ""
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    # Thresholds
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None

    @property
    def status(self) -> str:
        """Get status based on thresholds."""
        if self.error_threshold and self.value >= self.error_threshold:
            return "error"
        if self.warning_threshold and self.value >= self.warning_threshold:
            return "warning"
        return "ok"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": self.type.value,
            "value": self.value,
            "unit": self.unit,
            "file_path": self.file_path,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "details": self.details,
            "status": self.status,
        }


@dataclass
class HotspotFinding:
    """A complexity hotspot in the codebase."""

    file_path: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    start_line: int = 1
    end_line: int = 1
    complexity: float = 0.0
    lines_of_code: int = 0
    cognitive_complexity: Optional[float] = None
    change_frequency: int = 0  # Number of commits touching this code
    last_modified: Optional[datetime] = None
    contributors: List[str] = field(default_factory=list)

    @property
    def risk_score(self) -> float:
        """Calculate risk score combining complexity and change frequency."""
        complexity_factor = min(self.complexity / 10, 1.0) if self.complexity else 0
        change_factor = min(self.change_frequency / 50, 1.0)
        return (complexity_factor * 0.7 + change_factor * 0.3) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "file_path": self.file_path,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "complexity": self.complexity,
            "lines_of_code": self.lines_of_code,
            "cognitive_complexity": self.cognitive_complexity,
            "change_frequency": self.change_frequency,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "contributors": self.contributors,
            "risk_score": self.risk_score,
        }


class SecretType(str, Enum):
    """Type of detected secret."""

    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    GITHUB_TOKEN = "github_token"
    GITHUB_PAT = "github_pat"
    GITLAB_TOKEN = "gitlab_token"
    SLACK_TOKEN = "slack_token"
    SLACK_WEBHOOK = "slack_webhook"
    DISCORD_TOKEN = "discord_token"
    DISCORD_WEBHOOK = "discord_webhook"
    STRIPE_KEY = "stripe_key"
    TWILIO_KEY = "twilio_key"
    SENDGRID_KEY = "sendgrid_key"
    MAILGUN_KEY = "mailgun_key"
    JWT_TOKEN = "jwt_token"
    PRIVATE_KEY = "private_key"
    GOOGLE_API_KEY = "google_api_key"
    AZURE_KEY = "azure_key"
    OPENAI_KEY = "openai_key"
    ANTHROPIC_KEY = "anthropic_key"
    DATABASE_URL = "database_url"
    GENERIC_API_KEY = "generic_api_key"
    GENERIC_SECRET = "generic_secret"
    HIGH_ENTROPY = "high_entropy"


@dataclass
class SecretFinding:
    """A detected secret or credential in the codebase."""

    id: str
    secret_type: SecretType
    file_path: str
    line_number: int
    column_start: int
    column_end: int
    matched_text: str  # Redacted version (first/last 4 chars visible)
    context_line: str  # The full line with secret redacted
    severity: VulnerabilitySeverity
    confidence: float  # 0.0 to 1.0
    entropy: Optional[float] = None  # Shannon entropy if calculated
    commit_sha: Optional[str] = None  # If found in git history
    commit_author: Optional[str] = None
    commit_date: Optional[datetime] = None
    is_in_history: bool = False  # True if found in git history (not current)
    verified: bool = False  # True if verified as active credential
    remediation: Optional[str] = None

    @staticmethod
    def redact_secret(secret: str) -> str:
        """Redact a secret, showing only first and last 4 chars."""
        if len(secret) <= 8:
            return "*" * len(secret)
        return f"{secret[:4]}{'*' * (len(secret) - 8)}{secret[-4:]}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "secret_type": self.secret_type.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_start": self.column_start,
            "column_end": self.column_end,
            "matched_text": self.matched_text,
            "context_line": self.context_line,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "commit_sha": self.commit_sha,
            "commit_author": self.commit_author,
            "commit_date": self.commit_date.isoformat() if self.commit_date else None,
            "is_in_history": self.is_in_history,
            "verified": self.verified,
            "remediation": self.remediation,
        }


@dataclass
class SecretsScanResult:
    """Result of a secrets scan."""

    scan_id: str
    repository: str
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "running"
    error: Optional[str] = None
    files_scanned: int = 0
    secrets: List[SecretFinding] = field(default_factory=list)
    scanned_history: bool = False
    history_depth: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for s in self.secrets if s.severity == VulnerabilitySeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for s in self.secrets if s.severity == VulnerabilitySeverity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for s in self.secrets if s.severity == VulnerabilitySeverity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for s in self.secrets if s.severity == VulnerabilitySeverity.LOW)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scan_id": self.scan_id,
            "repository": self.repository,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "error": self.error,
            "files_scanned": self.files_scanned,
            "secrets": [s.to_dict() for s in self.secrets],
            "scanned_history": self.scanned_history,
            "history_depth": self.history_depth,
            "summary": {
                "total_secrets": len(self.secrets),
                "critical_count": self.critical_count,
                "high_count": self.high_count,
                "medium_count": self.medium_count,
                "low_count": self.low_count,
                "current_files": sum(1 for s in self.secrets if not s.is_in_history),
                "in_history": sum(1 for s in self.secrets if s.is_in_history),
            },
        }
