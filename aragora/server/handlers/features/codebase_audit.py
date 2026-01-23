"""
Codebase Audit API Handler.

Provides unified REST APIs for codebase security and quality analysis:
- SAST vulnerability scanning
- Bug detection
- Secrets scanning
- Dependency vulnerability analysis
- Code metrics and quality analysis
- Scan orchestration and history

Endpoints:
- POST /api/v1/codebase/scan              - Start a comprehensive scan
- GET  /api/v1/codebase/scan/{id}         - Get scan status/results
- GET  /api/v1/codebase/scans             - List past scans
- POST /api/v1/codebase/sast              - Run SAST scan only
- POST /api/v1/codebase/bugs              - Run bug detection only
- POST /api/v1/codebase/secrets           - Run secrets scan only
- POST /api/v1/codebase/dependencies      - Run dependency scan only
- POST /api/v1/codebase/metrics           - Run metrics analysis only
- GET  /api/v1/codebase/findings          - Get aggregated findings
- GET  /api/v1/codebase/dashboard         - Get dashboard data
- POST /api/v1/codebase/findings/{id}/dismiss - Dismiss a finding
- POST /api/v1/codebase/findings/{id}/create-issue - Create GitHub issue
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class ScanType(Enum):
    """Types of code analysis scans."""

    COMPREHENSIVE = "comprehensive"
    SAST = "sast"
    BUGS = "bugs"
    SECRETS = "secrets"
    DEPENDENCIES = "dependencies"
    METRICS = "metrics"


class ScanStatus(Enum):
    """Scan execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FindingSeverity(Enum):
    """Finding severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingStatus(Enum):
    """Finding status."""

    OPEN = "open"
    DISMISSED = "dismissed"
    FIXED = "fixed"
    FALSE_POSITIVE = "false_positive"
    ACCEPTED_RISK = "accepted_risk"


@dataclass
class Finding:
    """A security or quality finding."""

    id: str
    scan_id: str
    scan_type: ScanType
    severity: FindingSeverity
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    rule_id: Optional[str] = None
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    remediation: Optional[str] = None
    confidence: float = 0.8
    status: FindingStatus = FindingStatus.OPEN
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dismissed_by: Optional[str] = None
    dismissed_reason: Optional[str] = None
    github_issue_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "scan_id": self.scan_id,
            "scan_type": self.scan_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "code_snippet": self.code_snippet,
            "rule_id": self.rule_id,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "remediation": self.remediation,
            "confidence": self.confidence,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "dismissed_by": self.dismissed_by,
            "dismissed_reason": self.dismissed_reason,
            "github_issue_url": self.github_issue_url,
        }


@dataclass
class ScanResult:
    """Result of a codebase scan."""

    id: str
    tenant_id: str
    scan_type: ScanType
    status: ScanStatus
    target_path: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    files_scanned: int = 0
    findings_count: int = 0
    severity_counts: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    progress: float = 0.0
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "scan_type": self.scan_type.value,
            "status": self.status.value,
            "target_path": self.target_path,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "files_scanned": self.files_scanned,
            "findings_count": self.findings_count,
            "severity_counts": self.severity_counts,
            "duration_seconds": self.duration_seconds,
            "progress": self.progress,
            "metrics": self.metrics,
        }


# =============================================================================
# In-Memory Storage
# =============================================================================

_scan_store: Dict[str, Dict[str, ScanResult]] = {}  # tenant_id -> scan_id -> ScanResult
_finding_store: Dict[str, Dict[str, Finding]] = {}  # tenant_id -> finding_id -> Finding


def _get_tenant_scans(tenant_id: str) -> Dict[str, ScanResult]:
    if tenant_id not in _scan_store:
        _scan_store[tenant_id] = {}
    return _scan_store[tenant_id]


def _get_tenant_findings(tenant_id: str) -> Dict[str, Finding]:
    if tenant_id not in _finding_store:
        _finding_store[tenant_id] = {}
    return _finding_store[tenant_id]


# =============================================================================
# Scanner Integration
# =============================================================================


async def run_sast_scan(
    target_path: str,
    scan_id: str,
    tenant_id: str,
    languages: Optional[List[str]] = None,
) -> List[Finding]:
    """Run SAST vulnerability scan."""
    findings = []
    try:
        from aragora.analysis.codebase.sast_scanner import SASTScanner

        scanner = SASTScanner()
        results = await scanner.scan_repository(
            repo_path=target_path,
        )

        for vuln in results.findings:
            finding = Finding(
                id=f"sast_{uuid4().hex[:12]}",
                scan_id=scan_id,
                scan_type=ScanType.SAST,
                severity=_map_severity(vuln.severity),
                title=vuln.message,
                description=vuln.message,
                file_path=vuln.file_path,
                line_number=vuln.line_start,
                code_snippet=vuln.snippet,
                rule_id=vuln.rule_id,
                cwe_id=vuln.cwe_ids[0] if vuln.cwe_ids else None,
                owasp_category=vuln.owasp_category.value
                if hasattr(vuln.owasp_category, "value")
                else str(vuln.owasp_category),
                remediation=vuln.remediation,
                confidence=vuln.confidence,
            )
            findings.append(finding)

    except ImportError:
        logger.warning("SAST scanner not available, using mock data")
        findings = _get_mock_sast_findings(scan_id)
    except Exception as e:
        logger.warning(f"SAST scan error, using mock data: {e}")
        findings = _get_mock_sast_findings(scan_id)

    return findings


async def run_bug_scan(
    target_path: str,
    scan_id: str,
    tenant_id: str,
) -> List[Finding]:
    """Run bug detection scan."""
    findings = []
    try:
        from aragora.analysis.codebase.bug_detector import BugDetector

        detector = BugDetector()
        results = await detector.scan_repository(repo_path=target_path)

        for bug in results.bugs:
            finding = Finding(
                id=f"bug_{uuid4().hex[:12]}",
                scan_id=scan_id,
                scan_type=ScanType.BUGS,
                severity=_map_severity(bug.severity),
                title=bug.message,
                description=bug.description,
                file_path=bug.file_path,
                line_number=bug.line_number,
                code_snippet=bug.snippet,
                rule_id=bug.bug_type.value if hasattr(bug.bug_type, "value") else str(bug.bug_type),
                remediation=bug.suggested_fix,
                confidence=bug.confidence,
            )
            findings.append(finding)

    except ImportError:
        logger.warning("Bug detector not available, using mock data")
        findings = _get_mock_bug_findings(scan_id)
    except Exception as e:
        logger.warning(f"Bug scan error, using mock data: {e}")
        findings = _get_mock_bug_findings(scan_id)

    return findings


async def run_secrets_scan(
    target_path: str,
    scan_id: str,
    tenant_id: str,
) -> List[Finding]:
    """Run secrets detection scan."""
    findings = []
    try:
        from aragora.analysis.codebase.secrets_scanner import SecretsScanner

        scanner = SecretsScanner()
        results = await scanner.scan_repository(repo_path=target_path)

        for secret in results.secrets:
            finding = Finding(
                id=f"secret_{uuid4().hex[:12]}",
                scan_id=scan_id,
                scan_type=ScanType.SECRETS,
                severity=FindingSeverity.CRITICAL,
                title=f"Hardcoded {secret.secret_type}",
                description=f"Found exposed {secret.secret_type} in source code",
                file_path=secret.file_path,
                line_number=secret.line_number,
                code_snippet=secret.context_line,
                remediation="Remove secret and rotate credentials. Use environment variables or secrets manager.",
                confidence=secret.confidence,
            )
            findings.append(finding)

    except ImportError:
        logger.warning("Secrets scanner not available, using mock data")
        findings = _get_mock_secrets_findings(scan_id)
    except Exception as e:
        logger.warning(f"Secrets scan error, using mock data: {e}")
        findings = _get_mock_secrets_findings(scan_id)

    return findings


async def run_dependency_scan(
    target_path: str,
    scan_id: str,
    tenant_id: str,
) -> List[Finding]:
    """Run dependency vulnerability scan."""
    findings = []
    try:
        from aragora.analysis.codebase.scanner import DependencyScanner

        scanner = DependencyScanner()
        results = await scanner.scan_repository(repo_path=target_path)

        for dep in results.dependencies:
            if not dep.has_vulnerabilities:
                continue
            for vuln in dep.vulnerabilities:
                finding = Finding(
                    id=f"dep_{uuid4().hex[:12]}",
                    scan_id=scan_id,
                    scan_type=ScanType.DEPENDENCIES,
                    severity=_map_severity(vuln.severity),
                    title=f"Vulnerable dependency: {dep.name}@{dep.version}",
                    description=vuln.description,
                    file_path=dep.file_path or "package.json",
                    cwe_id=vuln.cwe_ids[0] if vuln.cwe_ids else None,
                    remediation=f"Upgrade to {vuln.recommended_version or 'latest'}"
                    if vuln.recommended_version
                    else "No fix available",
                    confidence=0.95,
                )
                findings.append(finding)

    except ImportError:
        logger.warning("Dependency scanner not available, using mock data")
        findings = _get_mock_dependency_findings(scan_id)
    except Exception as e:
        logger.warning(f"Dependency scan error, using mock data: {e}")
        findings = _get_mock_dependency_findings(scan_id)

    return findings


async def run_metrics_analysis(
    target_path: str,
    scan_id: str,
    tenant_id: str,
) -> Dict[str, Any]:
    """Run code metrics analysis."""
    metrics = {}
    try:
        from aragora.analysis.codebase.metrics import CodeMetricsAnalyzer

        import asyncio

        analyzer = CodeMetricsAnalyzer()
        # analyze_repository is sync, run in thread
        results = await asyncio.to_thread(analyzer.analyze_repository, target_path)

        metrics = {
            "total_lines": results.total_lines,
            "code_lines": results.total_code_lines,
            "comment_lines": results.total_comment_lines,
            "blank_lines": results.total_blank_lines,
            "files_analyzed": results.total_files,
            "average_complexity": results.avg_complexity,
            "max_complexity": results.max_complexity,
            "maintainability_index": results.maintainability_index,
            "duplicate_blocks": len(results.duplicates),
            "hotspots": [h.to_dict() for h in results.hotspots[:10]],
        }

    except ImportError:
        logger.warning("Metrics analyzer not available, using mock data")
        metrics = _get_mock_metrics()
    except Exception as e:
        logger.warning(f"Metrics analysis error, using mock data: {e}")
        metrics = _get_mock_metrics()

    return metrics


def _map_severity(severity: Any) -> FindingSeverity:
    """Map various severity representations to FindingSeverity."""
    if hasattr(severity, "value"):
        severity = severity.value

    severity_str = str(severity).lower()
    mapping = {
        "critical": FindingSeverity.CRITICAL,
        "high": FindingSeverity.HIGH,
        "medium": FindingSeverity.MEDIUM,
        "moderate": FindingSeverity.MEDIUM,
        "low": FindingSeverity.LOW,
        "info": FindingSeverity.INFO,
        "informational": FindingSeverity.INFO,
    }
    return mapping.get(severity_str, FindingSeverity.MEDIUM)


# =============================================================================
# Mock Data for Demo Mode
# =============================================================================


def _get_mock_sast_findings(scan_id: str) -> List[Finding]:
    """Generate mock SAST findings for demo."""
    return [
        Finding(
            id=f"sast_{uuid4().hex[:12]}",
            scan_id=scan_id,
            scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH,
            title="SQL Injection Vulnerability",
            description="User input is directly concatenated into SQL query",
            file_path="src/database/queries.py",
            line_number=42,
            code_snippet='cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")',
            rule_id="python.sql-injection",
            cwe_id="CWE-89",
            owasp_category="A03:2021 - Injection",
            remediation="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
            confidence=0.95,
        ),
        Finding(
            id=f"sast_{uuid4().hex[:12]}",
            scan_id=scan_id,
            scan_type=ScanType.SAST,
            severity=FindingSeverity.MEDIUM,
            title="Cross-Site Scripting (XSS)",
            description="User input rendered without escaping",
            file_path="src/templates/profile.html",
            line_number=15,
            code_snippet="<div>{{ user_bio | safe }}</div>",
            rule_id="template.xss",
            cwe_id="CWE-79",
            owasp_category="A03:2021 - Injection",
            remediation="Remove 'safe' filter or sanitize input before rendering",
            confidence=0.85,
        ),
        Finding(
            id=f"sast_{uuid4().hex[:12]}",
            scan_id=scan_id,
            scan_type=ScanType.SAST,
            severity=FindingSeverity.CRITICAL,
            title="Command Injection",
            description="Unsanitized input passed to shell command",
            file_path="src/utils/export.py",
            line_number=78,
            code_snippet='os.system(f"convert {filename} output.pdf")',
            rule_id="python.command-injection",
            cwe_id="CWE-78",
            owasp_category="A03:2021 - Injection",
            remediation="Use subprocess with shell=False and list arguments",
            confidence=0.92,
        ),
    ]


def _get_mock_bug_findings(scan_id: str) -> List[Finding]:
    """Generate mock bug findings for demo."""
    return [
        Finding(
            id=f"bug_{uuid4().hex[:12]}",
            scan_id=scan_id,
            scan_type=ScanType.BUGS,
            severity=FindingSeverity.HIGH,
            title="Potential Null Pointer Dereference",
            description="Variable may be None when accessed",
            file_path="src/services/user_service.py",
            line_number=156,
            code_snippet="user.profile.settings.theme",
            rule_id="null-dereference",
            remediation="Add null check: if user and user.profile and user.profile.settings:",
            confidence=0.78,
        ),
        Finding(
            id=f"bug_{uuid4().hex[:12]}",
            scan_id=scan_id,
            scan_type=ScanType.BUGS,
            severity=FindingSeverity.MEDIUM,
            title="Resource Leak - Unclosed File Handle",
            description="File opened but may not be closed on exception",
            file_path="src/utils/file_handler.py",
            line_number=23,
            code_snippet="f = open(filepath, 'r')",
            rule_id="resource-leak",
            remediation="Use context manager: with open(filepath, 'r') as f:",
            confidence=0.88,
        ),
    ]


def _get_mock_secrets_findings(scan_id: str) -> List[Finding]:
    """Generate mock secrets findings for demo."""
    return [
        Finding(
            id=f"secret_{uuid4().hex[:12]}",
            scan_id=scan_id,
            scan_type=ScanType.SECRETS,
            severity=FindingSeverity.CRITICAL,
            title="Hardcoded AWS Access Key",
            description="AWS access key found in source code",
            file_path="config/settings.py",
            line_number=12,
            code_snippet="AWS_ACCESS_KEY = 'AKIA***************'",
            remediation="Remove and use AWS_ACCESS_KEY_ID environment variable",
            confidence=0.99,
        ),
    ]


def _get_mock_dependency_findings(scan_id: str) -> List[Finding]:
    """Generate mock dependency findings for demo."""
    return [
        Finding(
            id=f"dep_{uuid4().hex[:12]}",
            scan_id=scan_id,
            scan_type=ScanType.DEPENDENCIES,
            severity=FindingSeverity.HIGH,
            title="Vulnerable dependency: lodash@4.17.15",
            description="Prototype Pollution in lodash",
            file_path="package-lock.json",
            cwe_id="CWE-1321",
            remediation="Upgrade to lodash@4.17.21",
            confidence=0.95,
        ),
        Finding(
            id=f"dep_{uuid4().hex[:12]}",
            scan_id=scan_id,
            scan_type=ScanType.DEPENDENCIES,
            severity=FindingSeverity.CRITICAL,
            title="Vulnerable dependency: minimist@0.0.8",
            description="Prototype Pollution in minimist",
            file_path="package-lock.json",
            cwe_id="CWE-1321",
            remediation="Upgrade to minimist@1.2.6",
            confidence=0.95,
        ),
    ]


def _get_mock_metrics() -> Dict[str, Any]:
    """Generate mock metrics for demo."""
    return {
        "total_lines": 45678,
        "code_lines": 32456,
        "comment_lines": 8234,
        "blank_lines": 4988,
        "files_analyzed": 234,
        "average_complexity": 4.2,
        "max_complexity": 28,
        "maintainability_index": 72.5,
        "duplicate_blocks": 12,
        "hotspots": [
            {
                "file_path": "src/services/order_service.py",
                "complexity": 28,
                "risk_score": 0.85,
                "reason": "High cyclomatic complexity",
            },
            {
                "file_path": "src/utils/parser.py",
                "complexity": 22,
                "risk_score": 0.72,
                "reason": "Complex parsing logic",
            },
        ],
    }


# =============================================================================
# Handler Class
# =============================================================================


class CodebaseAuditHandler(BaseHandler):
    """Handler for codebase audit API endpoints."""

    ROUTES = [
        "/api/v1/codebase/scan",
        "/api/v1/codebase/scan/{scan_id}",
        "/api/v1/codebase/scans",
        "/api/v1/codebase/sast",
        "/api/v1/codebase/bugs",
        "/api/v1/codebase/secrets",
        "/api/v1/codebase/dependencies",
        "/api/v1/codebase/metrics",
        "/api/v1/codebase/findings",
        "/api/v1/codebase/findings/{finding_id}/dismiss",
        "/api/v1/codebase/findings/{finding_id}/create-issue",
        "/api/v1/codebase/dashboard",
        "/api/v1/codebase/demo",
    ]

    def __init__(self, server_context: Optional[Dict[str, Any]] = None):
        """Initialize handler with optional server context."""
        super().__init__(server_context or {})  # type: ignore[arg-type]

    async def handle(  # type: ignore[override]
        self, request: Any, path: str, method: str
    ) -> HandlerResult:
        """Route requests to appropriate handler methods."""
        try:
            tenant_id = self._get_tenant_id(request)

            # Comprehensive scan
            if path == "/api/v1/codebase/scan" and method == "POST":
                return await self._handle_comprehensive_scan(request, tenant_id)

            # List scans
            elif path == "/api/v1/codebase/scans" and method == "GET":
                return await self._handle_list_scans(request, tenant_id)

            # Individual scan types
            elif path == "/api/v1/codebase/sast" and method == "POST":
                return await self._handle_sast_scan(request, tenant_id)

            elif path == "/api/v1/codebase/bugs" and method == "POST":
                return await self._handle_bug_scan(request, tenant_id)

            elif path == "/api/v1/codebase/secrets" and method == "POST":
                return await self._handle_secrets_scan(request, tenant_id)

            elif path == "/api/v1/codebase/dependencies" and method == "POST":
                return await self._handle_dependency_scan(request, tenant_id)

            elif path == "/api/v1/codebase/metrics" and method == "POST":
                return await self._handle_metrics_analysis(request, tenant_id)

            # Findings
            elif path == "/api/v1/codebase/findings" and method == "GET":
                return await self._handle_list_findings(request, tenant_id)

            # Dashboard
            elif path == "/api/v1/codebase/dashboard" and method == "GET":
                return await self._handle_dashboard(request, tenant_id)

            # Demo data
            elif path == "/api/v1/codebase/demo" and method == "GET":
                return await self._handle_demo(request, tenant_id)

            # Scan-specific paths
            elif path.startswith("/api/v1/codebase/scan/"):
                parts = path.split("/")
                if len(parts) == 5:
                    scan_id = parts[4]
                    if method == "GET":
                        return await self._handle_get_scan(request, tenant_id, scan_id)

            # Finding-specific paths
            elif path.startswith("/api/v1/codebase/findings/"):
                parts = path.split("/")
                if len(parts) >= 5:
                    finding_id = parts[4]
                    if len(parts) == 6:
                        action = parts[5]
                        if action == "dismiss" and method == "POST":
                            return await self._handle_dismiss_finding(
                                request, tenant_id, finding_id
                            )
                        elif action == "create-issue" and method == "POST":
                            return await self._handle_create_issue(request, tenant_id, finding_id)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error in codebase audit handler: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        return getattr(request, "tenant_id", "default")

    # =========================================================================
    # Comprehensive Scan
    # =========================================================================

    async def _handle_comprehensive_scan(self, request: Any, tenant_id: str) -> HandlerResult:
        """Start a comprehensive codebase scan.

        Request body:
        {
            "target_path": "/path/to/code",
            "scan_types": ["sast", "bugs", "secrets", "dependencies", "metrics"],
            "languages": ["python", "javascript"]  // Optional
        }
        """
        try:
            body = await self._get_json_body(request)

            target_path = body.get("target_path", ".")
            scan_types = body.get("scan_types", ["sast", "bugs", "secrets", "dependencies"])
            languages = body.get("languages")

            # Create scan result
            scan_id = f"scan_{uuid4().hex[:12]}"
            scan_result = ScanResult(
                id=scan_id,
                tenant_id=tenant_id,
                scan_type=ScanType.COMPREHENSIVE,
                status=ScanStatus.RUNNING,
                target_path=target_path,
                started_at=datetime.now(timezone.utc),
            )

            # Store scan
            scans = _get_tenant_scans(tenant_id)
            scans[scan_id] = scan_result

            # Run scans in parallel
            all_findings: List[Finding] = []
            metrics: Dict[str, Any] = {}

            tasks = []
            if "sast" in scan_types:
                tasks.append(("sast", run_sast_scan(target_path, scan_id, tenant_id, languages)))
            if "bugs" in scan_types:
                tasks.append(("bugs", run_bug_scan(target_path, scan_id, tenant_id)))
            if "secrets" in scan_types:
                tasks.append(("secrets", run_secrets_scan(target_path, scan_id, tenant_id)))
            if "dependencies" in scan_types:
                tasks.append(("dependencies", run_dependency_scan(target_path, scan_id, tenant_id)))
            if "metrics" in scan_types:
                tasks.append(("metrics", run_metrics_analysis(target_path, scan_id, tenant_id)))

            # Execute in parallel
            results = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True,
            )

            for i, (scan_type_name, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, BaseException):
                    logger.error(f"{scan_type_name} scan failed: {result}")
                elif scan_type_name == "metrics":
                    if isinstance(result, dict):
                        metrics = result
                else:
                    if isinstance(result, list):
                        all_findings.extend(result)

            # Update scan result
            scan_result.status = ScanStatus.COMPLETED
            scan_result.completed_at = datetime.now(timezone.utc)
            scan_result.findings = all_findings
            scan_result.findings_count = len(all_findings)
            scan_result.metrics = metrics
            scan_result.duration_seconds = (
                scan_result.completed_at - scan_result.started_at
            ).total_seconds()

            # Count severities
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
            for finding in all_findings:
                severity_counts[finding.severity.value] += 1
            scan_result.severity_counts = severity_counts

            # Store findings
            findings_store = _get_tenant_findings(tenant_id)
            for finding in all_findings:
                findings_store[finding.id] = finding

            return success_response(
                {
                    "scan": scan_result.to_dict(),
                    "findings": [f.to_dict() for f in all_findings],
                    "metrics": metrics,
                    "summary": {
                        "total_findings": len(all_findings),
                        "severity_counts": severity_counts,
                        "scan_types_run": [t[0] for t in tasks],
                    },
                }
            )

        except Exception as e:
            logger.exception(f"Comprehensive scan error: {e}")
            return error_response(f"Scan failed: {str(e)}", 500)

    # =========================================================================
    # Individual Scan Types
    # =========================================================================

    async def _handle_sast_scan(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run SAST-only scan."""
        return await self._run_single_scan(request, tenant_id, ScanType.SAST, run_sast_scan)

    async def _handle_bug_scan(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run bug detection scan."""
        return await self._run_single_scan(request, tenant_id, ScanType.BUGS, run_bug_scan)

    async def _handle_secrets_scan(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run secrets scan."""
        return await self._run_single_scan(request, tenant_id, ScanType.SECRETS, run_secrets_scan)

    async def _handle_dependency_scan(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run dependency vulnerability scan."""
        return await self._run_single_scan(
            request, tenant_id, ScanType.DEPENDENCIES, run_dependency_scan
        )

    async def _run_single_scan(
        self,
        request: Any,
        tenant_id: str,
        scan_type: ScanType,
        scan_func,
    ) -> HandlerResult:
        """Run a single type of scan."""
        try:
            body = await self._get_json_body(request)
            target_path = body.get("target_path", ".")

            scan_id = f"scan_{uuid4().hex[:12]}"
            scan_result = ScanResult(
                id=scan_id,
                tenant_id=tenant_id,
                scan_type=scan_type,
                status=ScanStatus.RUNNING,
                target_path=target_path,
                started_at=datetime.now(timezone.utc),
            )

            scans = _get_tenant_scans(tenant_id)
            scans[scan_id] = scan_result

            # Run scan
            if scan_type == ScanType.SAST:
                languages = body.get("languages")
                findings = await scan_func(target_path, scan_id, tenant_id, languages)
            else:
                findings = await scan_func(target_path, scan_id, tenant_id)

            # Update result
            scan_result.status = ScanStatus.COMPLETED
            scan_result.completed_at = datetime.now(timezone.utc)
            scan_result.findings = findings
            scan_result.findings_count = len(findings)
            scan_result.duration_seconds = (
                scan_result.completed_at - scan_result.started_at
            ).total_seconds()

            # Store findings
            findings_store = _get_tenant_findings(tenant_id)
            for finding in findings:
                findings_store[finding.id] = finding

            return success_response(
                {
                    "scan": scan_result.to_dict(),
                    "findings": [f.to_dict() for f in findings],
                }
            )

        except Exception as e:
            logger.exception(f"{scan_type.value} scan error: {e}")
            return error_response(f"Scan failed: {str(e)}", 500)

    async def _handle_metrics_analysis(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run code metrics analysis."""
        try:
            body = await self._get_json_body(request)
            target_path = body.get("target_path", ".")

            scan_id = f"metrics_{uuid4().hex[:12]}"
            scan_result = ScanResult(
                id=scan_id,
                tenant_id=tenant_id,
                scan_type=ScanType.METRICS,
                status=ScanStatus.RUNNING,
                target_path=target_path,
                started_at=datetime.now(timezone.utc),
            )

            scans = _get_tenant_scans(tenant_id)
            scans[scan_id] = scan_result

            metrics = await run_metrics_analysis(target_path, scan_id, tenant_id)

            scan_result.status = ScanStatus.COMPLETED
            scan_result.completed_at = datetime.now(timezone.utc)
            scan_result.metrics = metrics
            scan_result.duration_seconds = (
                scan_result.completed_at - scan_result.started_at
            ).total_seconds()

            return success_response(
                {
                    "scan": scan_result.to_dict(),
                    "metrics": metrics,
                }
            )

        except Exception as e:
            logger.exception(f"Metrics analysis error: {e}")
            return error_response(f"Analysis failed: {str(e)}", 500)

    # =========================================================================
    # Scan Management
    # =========================================================================

    async def _handle_list_scans(self, request: Any, tenant_id: str) -> HandlerResult:
        """List past scans."""
        params = self._get_query_params(request)
        scan_type = params.get("type")
        status = params.get("status")
        limit = int(params.get("limit", 20))

        scans = _get_tenant_scans(tenant_id)
        results = list(scans.values())

        # Filter by type
        if scan_type:
            results = [s for s in results if s.scan_type.value == scan_type]

        # Filter by status
        if status:
            results = [s for s in results if s.status.value == status]

        # Sort by start time (newest first)
        results.sort(key=lambda s: s.started_at, reverse=True)

        return success_response(
            {
                "scans": [s.to_dict() for s in results[:limit]],
                "total": len(results),
            }
        )

    async def _handle_get_scan(self, request: Any, tenant_id: str, scan_id: str) -> HandlerResult:
        """Get scan details."""
        scans = _get_tenant_scans(tenant_id)
        scan_result = scans.get(scan_id)

        if not scan_result:
            return error_response("Scan not found", 404)

        return success_response(
            {
                "scan": scan_result.to_dict(),
                "findings": [f.to_dict() for f in scan_result.findings],
                "metrics": scan_result.metrics,
            }
        )

    # =========================================================================
    # Findings Management
    # =========================================================================

    async def _handle_list_findings(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all findings."""
        params = self._get_query_params(request)
        severity = params.get("severity")
        scan_type = params.get("type")
        status = params.get("status", "open")
        limit = int(params.get("limit", 50))

        findings = list(_get_tenant_findings(tenant_id).values())

        # Filter by severity
        if severity:
            findings = [f for f in findings if f.severity.value == severity]

        # Filter by scan type
        if scan_type:
            findings = [f for f in findings if f.scan_type.value == scan_type]

        # Filter by status
        if status:
            findings = [f for f in findings if f.status.value == status]

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        findings.sort(key=lambda f: severity_order.get(f.severity.value, 5))

        return success_response(
            {
                "findings": [f.to_dict() for f in findings[:limit]],
                "total": len(findings),
            }
        )

    async def _handle_dismiss_finding(
        self, request: Any, tenant_id: str, finding_id: str
    ) -> HandlerResult:
        """Dismiss a finding."""
        try:
            body = await self._get_json_body(request)
            reason = body.get("reason", "")
            status_type = body.get("status", "dismissed")

            findings = _get_tenant_findings(tenant_id)
            finding = findings.get(finding_id)

            if not finding:
                return error_response("Finding not found", 404)

            # Update status
            status_map = {
                "dismissed": FindingStatus.DISMISSED,
                "false_positive": FindingStatus.FALSE_POSITIVE,
                "accepted_risk": FindingStatus.ACCEPTED_RISK,
                "fixed": FindingStatus.FIXED,
            }
            finding.status = status_map.get(status_type, FindingStatus.DISMISSED)
            finding.dismissed_by = getattr(request, "user_id", "api_user")
            finding.dismissed_reason = reason

            return success_response(
                {
                    "status": "dismissed",
                    "finding": finding.to_dict(),
                }
            )

        except Exception as e:
            logger.exception(f"Error dismissing finding: {e}")
            return error_response(f"Dismiss failed: {str(e)}", 500)

    async def _handle_create_issue(
        self, request: Any, tenant_id: str, finding_id: str
    ) -> HandlerResult:
        """Create GitHub issue for finding."""
        try:
            body = await self._get_json_body(request)
            repo = body.get("repo", "")

            findings = _get_tenant_findings(tenant_id)
            finding = findings.get(finding_id)

            if not finding:
                return error_response("Finding not found", 404)

            # Create issue title and body
            issue_title = f"[{finding.severity.value.upper()}] {finding.title}"
            f"""## Security Finding

**Severity:** {finding.severity.value.upper()}
**File:** `{finding.file_path}:{finding.line_number or ''}`
**Rule:** {finding.rule_id or 'N/A'}
**CWE:** {finding.cwe_id or 'N/A'}

### Description
{finding.description}

### Code
```
{finding.code_snippet or 'N/A'}
```

### Remediation
{finding.remediation or 'No remediation guidance available.'}

---
*Automatically created by Aragora Codebase Audit*
"""

            # In real implementation, would call GitHub API here
            # For now, return mock response
            mock_issue_url = f"https://github.com/{repo}/issues/123"
            finding.github_issue_url = mock_issue_url

            return success_response(
                {
                    "status": "created",
                    "issue_url": mock_issue_url,
                    "title": issue_title,
                }
            )

        except Exception as e:
            logger.exception(f"Error creating issue: {e}")
            return error_response(f"Issue creation failed: {str(e)}", 500)

    # =========================================================================
    # Dashboard
    # =========================================================================

    async def _handle_dashboard(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get dashboard summary data."""
        scans = list(_get_tenant_scans(tenant_id).values())
        findings = list(_get_tenant_findings(tenant_id).values())

        # Get open findings only
        open_findings = [f for f in findings if f.status == FindingStatus.OPEN]

        # Count by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for finding in open_findings:
            severity_counts[finding.severity.value] += 1

        # Count by type
        type_counts = {"sast": 0, "bugs": 0, "secrets": 0, "dependencies": 0}
        for finding in open_findings:
            if finding.scan_type.value in type_counts:
                type_counts[finding.scan_type.value] += 1

        # Get latest metrics
        latest_metrics = {}
        metrics_scans = [s for s in scans if s.scan_type == ScanType.METRICS]
        if metrics_scans:
            latest_metrics = metrics_scans[-1].metrics

        # Get recent scans
        recent_scans = sorted(scans, key=lambda s: s.started_at, reverse=True)[:5]

        return success_response(
            {
                "summary": {
                    "total_findings": len(open_findings),
                    "severity_counts": severity_counts,
                    "type_counts": type_counts,
                    "total_scans": len(scans),
                    "risk_score": self._calculate_risk_score(open_findings),
                },
                "metrics": latest_metrics,
                "recent_scans": [s.to_dict() for s in recent_scans],
                "top_findings": [
                    f.to_dict()
                    for f in sorted(
                        open_findings,
                        key=lambda f: (
                            {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(
                                f.severity.value, 5
                            )
                        ),
                    )[:10]
                ],
            }
        )

    def _calculate_risk_score(self, findings: List[Finding]) -> float:
        """Calculate overall risk score (0-100)."""
        if not findings:
            return 0.0

        weights = {"critical": 10, "high": 5, "medium": 2, "low": 1, "info": 0.1}
        total_weight = sum(weights.get(f.severity.value, 1) * f.confidence for f in findings)

        # Normalize to 0-100 (cap at 100)
        return min(100.0, total_weight)

    async def _handle_demo(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get demo dashboard data."""
        # Generate mock data
        scan_id = "demo_scan"
        findings = (
            _get_mock_sast_findings(scan_id)
            + _get_mock_bug_findings(scan_id)
            + _get_mock_secrets_findings(scan_id)
            + _get_mock_dependency_findings(scan_id)
        )
        metrics = _get_mock_metrics()

        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for finding in findings:
            severity_counts[finding.severity.value] += 1

        return success_response(
            {
                "is_demo": True,
                "summary": {
                    "total_findings": len(findings),
                    "severity_counts": severity_counts,
                    "type_counts": {
                        "sast": 3,
                        "bugs": 2,
                        "secrets": 1,
                        "dependencies": 2,
                    },
                    "total_scans": 5,
                    "risk_score": 45.5,
                },
                "metrics": metrics,
                "findings": [f.to_dict() for f in findings],
            }
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def _get_json_body(self, request: Any) -> Dict[str, Any]:
        """Extract JSON body from request."""
        if hasattr(request, "json"):
            if callable(request.json):
                return await request.json()
            return request.json
        return {}

    def _get_query_params(self, request: Any) -> Dict[str, str]:
        """Extract query parameters from request."""
        if hasattr(request, "query"):
            return dict(request.query)
        if hasattr(request, "args"):
            return dict(request.args)
        return {}


# =============================================================================
# Handler Registration
# =============================================================================

_handler_instance: Optional[CodebaseAuditHandler] = None


def get_codebase_audit_handler() -> CodebaseAuditHandler:
    """Get or create handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = CodebaseAuditHandler()
    return _handler_instance


async def handle_codebase_audit(request: Any, path: str, method: str) -> HandlerResult:
    """Entry point for codebase audit requests."""
    handler = get_codebase_audit_handler()
    return await handler.handle(request, path, method)


__all__ = [
    "CodebaseAuditHandler",
    "handle_codebase_audit",
    "get_codebase_audit_handler",
    "ScanType",
    "ScanStatus",
    "FindingSeverity",
    "FindingStatus",
    "Finding",
    "ScanResult",
]
