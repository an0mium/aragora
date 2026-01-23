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
