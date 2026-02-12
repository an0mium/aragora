"""
Codebase Audit Scanning Logic.

Contains scanner integration functions (SAST, bugs, secrets, dependencies,
metrics) and mock data generators for demo mode.

Security: All scan functions perform defense-in-depth path validation
before passing paths to any scanner. This ensures safety even if callers
bypass handler-level validation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .rules import (
    Finding,
    FindingSeverity,
    ScanType,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Path Security
# =============================================================================

# Allowed language identifiers for SAST scanning (whitelist)
_ALLOWED_LANGUAGES = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "rust",
        "c",
        "cpp",
        "csharp",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "haskell",
        "elixir",
        "erlang",
        "lua",
        "perl",
        "r",
        "shell",
        "bash",
        "powershell",
        "sql",
        "html",
        "css",
    }
)

# Characters that must never appear in scan target paths
_DANGEROUS_PATH_CHARS = re.compile(r"[;\|&\$`\(\)\{\}<>\n\r\x00]")

# Maximum allowed path length
_MAX_PATH_LENGTH = 4096


class ScanPathError(ValueError):
    """Raised when a scan target path fails security validation."""


def _validate_scan_path(target_path: str) -> str:
    """Validate and normalize a scan target path.

    Performs defense-in-depth validation to prevent path traversal,
    command injection via path strings, and access to sensitive locations.

    Args:
        target_path: The user-supplied path to validate.

    Returns:
        The normalized, validated path string.

    Raises:
        ScanPathError: If the path fails any validation check.
    """
    if not isinstance(target_path, str):
        raise ScanPathError("target_path must be a string")

    # Strip whitespace
    target_path = target_path.strip()

    if not target_path:
        raise ScanPathError("target_path cannot be empty")

    # Length check to prevent DoS
    if len(target_path) > _MAX_PATH_LENGTH:
        raise ScanPathError(f"target_path exceeds maximum length of {_MAX_PATH_LENGTH} characters")

    # Block null bytes (used in path injection attacks)
    if "\x00" in target_path:
        raise ScanPathError("target_path contains null byte")

    # Block shell metacharacters that could enable injection if the path
    # is ever interpolated into a shell context downstream
    match = _DANGEROUS_PATH_CHARS.search(target_path)
    if match:
        raise ScanPathError(f"target_path contains disallowed character: {repr(match.group())}")

    # Block home directory expansion
    if target_path.startswith("~"):
        raise ScanPathError("Home directory expansion (~) is not allowed")

    # Block absolute paths -- only relative paths within the workspace are safe
    if os.path.isabs(target_path):
        raise ScanPathError("Absolute paths are not allowed; use relative paths only")

    # Normalize and resolve to catch traversal via symlinks or redundant separators.
    # We resolve relative to CWD so we can verify the result stays within CWD.
    cwd = Path.cwd().resolve()
    resolved = (cwd / target_path).resolve()

    # Ensure the resolved path is within the working directory (prevents
    # traversal via symlinks, encoded sequences, or redundant separators).
    try:
        resolved.relative_to(cwd)
    except ValueError:
        raise ScanPathError("target_path resolves outside the allowed workspace directory")

    # Return the original relative path (not the resolved absolute) so
    # downstream scanners operate relative to CWD as expected.
    return target_path


def _validate_languages(languages: list[str] | None) -> list[str] | None:
    """Validate language identifiers against the allowed whitelist.

    Args:
        languages: Optional list of language names.

    Returns:
        The validated list, or None.

    Raises:
        ScanPathError: If any language identifier is invalid.
    """
    if languages is None:
        return None

    if not isinstance(languages, list):
        raise ScanPathError("languages must be a list of strings")

    validated: list[str] = []
    for lang in languages:
        if not isinstance(lang, str):
            raise ScanPathError(f"Invalid language identifier: {lang!r}")
        normalized = lang.strip().lower()
        if normalized not in _ALLOWED_LANGUAGES:
            raise ScanPathError(
                f"Unsupported language: {lang!r}. Allowed: {sorted(_ALLOWED_LANGUAGES)}"
            )
        validated.append(normalized)

    return validated if validated else None


# =============================================================================
# In-Memory Storage
# =============================================================================

_scan_store: dict[str, dict[str, Any]] = {}  # tenant_id -> scan_id -> ScanResult
_finding_store: dict[str, dict[str, Finding]] = {}  # tenant_id -> finding_id -> Finding


def _get_tenant_scans(tenant_id: str) -> dict[str, Any]:
    if tenant_id not in _scan_store:
        _scan_store[tenant_id] = {}
    return _scan_store[tenant_id]


def _get_tenant_findings(tenant_id: str) -> dict[str, Finding]:
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
    languages: list[str] | None = None,
) -> list[Finding]:
    """Run SAST vulnerability scan."""
    # Defense-in-depth: validate path even if handler already checked
    target_path = _validate_scan_path(target_path)
    languages = _validate_languages(languages)

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
                owasp_category=(
                    vuln.owasp_category.value
                    if hasattr(vuln.owasp_category, "value")
                    else str(vuln.owasp_category)
                ),
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
) -> list[Finding]:
    """Run bug detection scan."""
    # Defense-in-depth: validate path even if handler already checked
    target_path = _validate_scan_path(target_path)

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
) -> list[Finding]:
    """Run secrets detection scan."""
    # Defense-in-depth: validate path even if handler already checked
    target_path = _validate_scan_path(target_path)

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
) -> list[Finding]:
    """Run dependency vulnerability scan."""
    # Defense-in-depth: validate path even if handler already checked
    target_path = _validate_scan_path(target_path)

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
                    remediation=(
                        f"Upgrade to {vuln.recommended_version or 'latest'}"
                        if vuln.recommended_version
                        else "No fix available"
                    ),
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
) -> dict[str, Any]:
    """Run code metrics analysis."""
    # Defense-in-depth: validate path even if handler already checked
    target_path = _validate_scan_path(target_path)

    metrics = {}
    try:
        from aragora.analysis.codebase.metrics import CodeMetricsAnalyzer

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


def _get_mock_sast_findings(scan_id: str) -> list[Finding]:
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
            code_snippet='query = "SELECT * FROM users WHERE id = " + user_id; cursor.execute(query)',
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


def _get_mock_bug_findings(scan_id: str) -> list[Finding]:
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


def _get_mock_secrets_findings(scan_id: str) -> list[Finding]:
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


def _get_mock_dependency_findings(scan_id: str) -> list[Finding]:
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


def _get_mock_metrics() -> dict[str, Any]:
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
