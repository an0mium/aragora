"""
HTTP API Handlers for Codebase Security Analysis.

Provides REST APIs for security vulnerability scanning:
- Trigger dependency vulnerability scans
- Query CVE databases
- Get scan results and history
- View vulnerability details

Endpoints:
- POST /api/v1/codebase/{repo}/scan - Trigger security scan
- GET /api/v1/codebase/{repo}/scan/latest - Get latest scan result
- GET /api/v1/codebase/{repo}/scan/{scan_id} - Get specific scan result
- GET /api/v1/codebase/{repo}/vulnerabilities - List all vulnerabilities
- GET /api/v1/cve/{cve_id} - Get CVE details
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from aragora.analysis.codebase import (
    CVEClient,
    DependencyScanner,
    ScanResult,
    SecretsScanner,
    SecretsScanResult,
    SASTScanner,
    SASTScanResult,
    SBOMGenerator,
    SBOMFormat,
    SBOMResult,
)
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)
from aragora.events.security_events import (
    SecurityEvent,
    SecurityEventType,
    SecuritySeverity,
    SecurityFinding,
    get_security_emitter,
)
from aragora.services import ServiceRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Service Registry Integration
# =============================================================================


def _get_scanner() -> DependencyScanner:
    """Get or create DependencyScanner from service registry."""
    registry = ServiceRegistry.get()
    if not registry.has(DependencyScanner):
        scanner = DependencyScanner()
        registry.register(DependencyScanner, scanner)
        logger.info("Registered DependencyScanner with service registry")
    return registry.resolve(DependencyScanner)


def _get_cve_client() -> CVEClient:
    """Get or create CVEClient from service registry."""
    registry = ServiceRegistry.get()
    if not registry.has(CVEClient):
        client = CVEClient()
        registry.register(CVEClient, client)
        logger.info("Registered CVEClient with service registry")
    return registry.resolve(CVEClient)


def _get_secrets_scanner() -> SecretsScanner:
    """Get or create SecretsScanner from service registry."""
    registry = ServiceRegistry.get()
    if not registry.has(SecretsScanner):
        scanner = SecretsScanner()
        registry.register(SecretsScanner, scanner)
        logger.info("Registered SecretsScanner with service registry")
    return registry.resolve(SecretsScanner)


def _get_sast_scanner() -> SASTScanner:
    """Get or create SASTScanner from service registry."""
    registry = ServiceRegistry.get()
    if not registry.has(SASTScanner):
        scanner = SASTScanner()
        registry.register(SASTScanner, scanner)
        logger.info("Registered SASTScanner with service registry")
    return registry.resolve(SASTScanner)


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

_scan_results: Dict[str, Dict[str, ScanResult]] = {}  # repo_id -> {scan_id -> result}
_scan_lock = threading.Lock()
_running_scans: Dict[str, asyncio.Task] = {}

# Secrets scan storage
_secrets_scan_results: Dict[str, Dict[str, SecretsScanResult]] = {}
_secrets_scan_lock = threading.Lock()
_running_secrets_scans: Dict[str, asyncio.Task] = {}

# SAST scan storage
_sast_scan_results: Dict[str, Dict[str, SASTScanResult]] = {}
_sast_scan_lock = threading.Lock()
_running_sast_scans: Dict[str, asyncio.Task] = {}


def _get_or_create_repo_scans(repo_id: str) -> Dict[str, ScanResult]:
    """Get or create scan storage for a repository."""
    with _scan_lock:
        if repo_id not in _scan_results:
            _scan_results[repo_id] = {}
        return _scan_results[repo_id]


# =============================================================================
# Scan Handlers
# =============================================================================


async def handle_scan_repository(
    repo_path: str,
    repo_id: Optional[str] = None,
    branch: Optional[str] = None,
    commit_sha: Optional[str] = None,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> HandlerResult:
    """
    Trigger a security scan for a repository.

    POST /api/v1/codebase/{repo}/scan
    {
        "repo_path": "/path/to/repo",
        "branch": "main",
        "commit_sha": "abc123"
    }
    """
    try:
        repo_id = repo_id or f"repo_{uuid.uuid4().hex[:12]}"
        scan_id = f"scan_{uuid.uuid4().hex[:12]}"

        # Check if scan already running
        if repo_id in _running_scans:
            task = _running_scans[repo_id]
            if not task.done():
                return error_response("Scan already in progress", 409)

        # Create initial scan result
        scan_result = ScanResult(
            scan_id=scan_id,
            repository=repo_id,
            branch=branch,
            commit_sha=commit_sha,
            status="running",
        )

        repo_scans = _get_or_create_repo_scans(repo_id)
        repo_scans[scan_id] = scan_result

        # Start async scan
        async def run_scan():
            try:
                scanner = _get_scanner()
                result = await scanner.scan_repository(
                    repo_path=repo_path,
                    branch=branch,
                    commit_sha=commit_sha,
                )

                # Update stored result
                with _scan_lock:
                    repo_scans[scan_id] = result
                    result.scan_id = scan_id

                logger.info(
                    f"[Security] Completed scan {scan_id} for {repo_id}: "
                    f"{result.summary.vulnerable_dependencies} vulnerable deps found"
                )

                # Emit security events for findings (triggers debate for critical findings)
                await _emit_scan_events(result, repo_id, scan_id, workspace_id)

            except Exception as e:
                logger.exception(f"Scan {scan_id} failed: {e}")
                with _scan_lock:
                    scan_result.status = "failed"
                    scan_result.error = str(e)
                    scan_result.completed_at = datetime.now(timezone.utc)

            finally:
                if repo_id in _running_scans:
                    del _running_scans[repo_id]

        # Create and store task
        task = asyncio.create_task(run_scan())
        _running_scans[repo_id] = task

        logger.info(f"[Security] Started scan {scan_id} for {repo_id}")

        return success_response({
            "scan_id": scan_id,
            "status": "running",
            "repository": repo_id,
        })

    except Exception as e:
        logger.exception(f"Failed to start scan: {e}")
        return error_response(str(e), 500)


async def handle_get_scan_status(
    repo_id: str,
    scan_id: Optional[str] = None,
) -> HandlerResult:
    """
    Get scan status/result.

    GET /api/v1/codebase/{repo}/scan/latest
    GET /api/v1/codebase/{repo}/scan/{scan_id}
    """
    try:
        repo_scans = _get_or_create_repo_scans(repo_id)

        if scan_id:
            # Get specific scan
            scan = repo_scans.get(scan_id)
            if not scan:
                return error_response("Scan not found", 404)
            return success_response({"scan_result": scan.to_dict()})
        else:
            # Get latest scan
            if not repo_scans:
                return error_response("No scans found for repository", 404)

            # Sort by start time and get latest
            latest = max(repo_scans.values(), key=lambda s: s.started_at)
            return success_response({"scan_result": latest.to_dict()})

    except Exception as e:
        logger.exception(f"Failed to get scan status: {e}")
        return error_response(str(e), 500)


async def handle_get_vulnerabilities(
    repo_id: str,
    severity: Optional[str] = None,
    package: Optional[str] = None,
    ecosystem: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> HandlerResult:
    """
    Get vulnerabilities from latest scan.

    GET /api/v1/codebase/{repo}/vulnerabilities
    Query params: severity, package, ecosystem, limit, offset
    """
    try:
        repo_scans = _get_or_create_repo_scans(repo_id)

        if not repo_scans:
            return error_response("No scans found for repository", 404)

        # Get latest completed scan
        completed_scans = [s for s in repo_scans.values() if s.status == "completed"]
        if not completed_scans:
            return error_response("No completed scans found", 404)

        latest = max(completed_scans, key=lambda s: s.started_at)

        # Collect all vulnerabilities
        vulnerabilities = []
        for dep in latest.dependencies:
            for vuln in dep.vulnerabilities:
                vuln_dict = vuln.to_dict()
                vuln_dict["package_name"] = dep.name
                vuln_dict["package_version"] = dep.version
                vuln_dict["package_ecosystem"] = dep.ecosystem
                vulnerabilities.append(vuln_dict)

        # Filter
        if severity:
            vulnerabilities = [v for v in vulnerabilities if v["severity"] == severity]
        if package:
            vulnerabilities = [
                v for v in vulnerabilities if package.lower() in v["package_name"].lower()
            ]
        if ecosystem:
            vulnerabilities = [v for v in vulnerabilities if v["package_ecosystem"] == ecosystem]

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}
        vulnerabilities.sort(key=lambda v: severity_order.get(v["severity"], 5))

        # Paginate
        total = len(vulnerabilities)
        vulnerabilities = vulnerabilities[offset : offset + limit]

        return success_response({
            "vulnerabilities": vulnerabilities,
            "total": total,
            "limit": limit,
            "offset": offset,
            "scan_id": latest.scan_id,
        })

    except Exception as e:
        logger.exception(f"Failed to get vulnerabilities: {e}")
        return error_response(str(e), 500)


async def handle_get_cve_details(
    cve_id: str,
) -> HandlerResult:
    """
    Get CVE details from vulnerability databases.

    GET /api/v1/cve/{cve_id}
    """
    try:
        client = CVEClient()
        vuln = await client.get_cve(cve_id)

        if not vuln:
            return error_response(f"CVE {cve_id} not found", 404)

        return success_response({"vulnerability": vuln.to_dict()})

    except Exception as e:
        logger.exception(f"Failed to get CVE details: {e}")
        return error_response(str(e), 500)


async def handle_query_package_vulnerabilities(
    package_name: str,
    ecosystem: str,
    version: Optional[str] = None,
) -> HandlerResult:
    """
    Query vulnerabilities for a specific package.

    GET /api/v1/codebase/package/{ecosystem}/{package}/vulnerabilities
    """
    try:
        client = CVEClient()
        vulnerabilities = await client.query_package(
            package_name=package_name,
            ecosystem=ecosystem,
            version=version,
        )

        return success_response({
            "package": package_name,
            "ecosystem": ecosystem,
            "version": version,
            "vulnerabilities": [v.to_dict() for v in vulnerabilities],
            "total": len(vulnerabilities),
        })

    except Exception as e:
        logger.exception(f"Failed to query package vulnerabilities: {e}")
        return error_response(str(e), 500)


async def handle_list_scans(
    repo_id: str,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> HandlerResult:
    """
    List scan history for a repository.

    GET /api/v1/codebase/{repo}/scans
    """
    try:
        repo_scans = _get_or_create_repo_scans(repo_id)

        scans = list(repo_scans.values())

        # Filter by status
        if status:
            scans = [s for s in scans if s.status == status]

        # Sort by start time descending
        scans.sort(key=lambda s: s.started_at, reverse=True)

        # Paginate
        total = len(scans)
        scans = scans[offset : offset + limit]

        return success_response({
            "scans": [
                {
                    "scan_id": s.scan_id,
                    "status": s.status,
                    "started_at": s.started_at.isoformat(),
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                    "summary": {
                        "total_dependencies": s.total_dependencies,
                        "vulnerable_dependencies": s.vulnerable_dependencies,
                        "critical_count": s.critical_count,
                        "high_count": s.high_count,
                        "medium_count": s.medium_count,
                        "low_count": s.low_count,
                    }
                    if s.status == "completed"
                    else None,
                }
                for s in scans
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        })

    except Exception as e:
        logger.exception(f"Failed to list scans: {e}")
        return error_response(str(e), 500)


# =============================================================================
# Security Event Emission
# =============================================================================


async def _emit_scan_events(
    result: ScanResult,
    repo_id: str,
    scan_id: str,
    workspace_id: Optional[str] = None,
) -> None:
    """
    Emit security events for scan findings.

    Automatically triggers multi-agent debates for critical vulnerabilities.
    """
    try:
        emitter = get_security_emitter()

        # Build findings from scan result
        findings = []
        for dep in result.dependencies:
            for vuln in dep.vulnerabilities:
                severity_map = {
                    "critical": SecuritySeverity.CRITICAL,
                    "high": SecuritySeverity.HIGH,
                    "medium": SecuritySeverity.MEDIUM,
                    "low": SecuritySeverity.LOW,
                }
                vuln_severity_str = (
                    vuln.severity.value.lower()
                    if hasattr(vuln.severity, "value")
                    else str(vuln.severity).lower()
                )
                severity = severity_map.get(vuln_severity_str, SecuritySeverity.MEDIUM)

                findings.append(
                    SecurityFinding(
                        id=vuln.id,
                        finding_type="vulnerability",
                        severity=severity,
                        title=vuln.title or vuln.cve_id or "Unknown",
                        description=vuln.description or "",
                        cve_id=vuln.cve_id,
                        package_name=dep.name,
                        package_version=dep.version,
                        recommendation=vuln.recommendation,
                        metadata={
                            "ecosystem": dep.ecosystem,
                            "cvss_score": getattr(vuln, "cvss_score", None),
                            "sources": getattr(vuln, "sources", []),
                        },
                    )
                )

        if not findings:
            logger.debug(f"[Security] No findings to emit for scan {scan_id}")
            return

        # Determine overall severity
        critical_count = sum(1 for f in findings if f.severity == SecuritySeverity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SecuritySeverity.HIGH)

        if critical_count > 0:
            overall_severity = SecuritySeverity.CRITICAL
            event_type = SecurityEventType.CRITICAL_VULNERABILITY
        elif high_count > 0:
            overall_severity = SecuritySeverity.HIGH
            event_type = SecurityEventType.VULNERABILITY_DETECTED
        else:
            overall_severity = SecuritySeverity.MEDIUM
            event_type = SecurityEventType.SCAN_COMPLETED

        # Emit scan completed event with findings
        # The emitter will auto-trigger debate for critical findings
        event = SecurityEvent(
            event_type=event_type,
            severity=overall_severity,
            repository=repo_id,
            scan_id=scan_id,
            workspace_id=workspace_id,
            findings=findings[:20],  # Limit to top 20 findings
        )

        await emitter.emit(event)

        logger.info(
            f"[Security] Emitted {event_type.value} event for scan {scan_id}: "
            f"{critical_count} critical, {high_count} high severity findings"
        )

    except Exception as e:
        logger.warning(f"[Security] Failed to emit scan events: {e}")


async def _emit_secrets_events(
    result: SecretsScanResult,
    repo_id: str,
    scan_id: str,
    workspace_id: Optional[str] = None,
) -> None:
    """
    Emit security events for secrets scan findings.

    Automatically triggers multi-agent debates for critical secrets.
    """
    try:
        emitter = get_security_emitter()

        # Build findings from scan result
        findings = []
        for secret in result.secrets:
            severity_map = {
                "critical": SecuritySeverity.CRITICAL,
                "high": SecuritySeverity.HIGH,
                "medium": SecuritySeverity.MEDIUM,
                "low": SecuritySeverity.LOW,
            }
            secret_severity_str = (
                secret.severity.value.lower()
                if hasattr(secret.severity, "value")
                else str(secret.severity).lower()
            )
            severity = severity_map.get(secret_severity_str, SecuritySeverity.HIGH)

            findings.append(
                SecurityFinding(
                    id=secret.id,
                    finding_type="secret",
                    severity=severity,
                    title=f"Exposed {secret.secret_type.value if hasattr(secret.secret_type, 'value') else secret.secret_type}",
                    description=f"Hardcoded credential detected in {secret.file_path}",
                    file_path=secret.file_path,
                    line_number=secret.line_number,
                    recommendation="Rotate the credential immediately and remove from codebase",
                    metadata={
                        "secret_type": secret.secret_type.value
                        if hasattr(secret.secret_type, "value")
                        else str(secret.secret_type),
                        "confidence": secret.confidence,
                        "is_in_history": getattr(secret, "is_in_history", False),
                    },
                )
            )

        if not findings:
            logger.debug(f"[Security] No secrets findings to emit for scan {scan_id}")
            return

        # Determine overall severity
        critical_count = sum(1 for f in findings if f.severity == SecuritySeverity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SecuritySeverity.HIGH)

        if critical_count > 0:
            overall_severity = SecuritySeverity.CRITICAL
            event_type = SecurityEventType.CRITICAL_SECRET
        elif high_count > 0:
            overall_severity = SecuritySeverity.HIGH
            event_type = SecurityEventType.SECRET_DETECTED
        else:
            overall_severity = SecuritySeverity.MEDIUM
            event_type = SecurityEventType.SCAN_COMPLETED

        # Emit secrets event with findings
        event = SecurityEvent(
            event_type=event_type,
            severity=overall_severity,
            repository=repo_id,
            scan_id=scan_id,
            workspace_id=workspace_id,
            findings=findings[:20],  # Limit to top 20 findings
        )

        await emitter.emit(event)

        logger.info(
            f"[Security] Emitted {event_type.value} event for secrets scan {scan_id}: "
            f"{critical_count} critical, {high_count} high severity findings"
        )

    except Exception as e:
        logger.warning(f"[Security] Failed to emit secrets scan events: {e}")


# =============================================================================
# Secrets Scan Helpers
# =============================================================================


def _get_or_create_secrets_scans(repo_id: str) -> Dict[str, SecretsScanResult]:
    """Get or create secrets scan storage for a repository."""
    with _secrets_scan_lock:
        if repo_id not in _secrets_scan_results:
            _secrets_scan_results[repo_id] = {}
        return _secrets_scan_results[repo_id]


# =============================================================================
# Secrets Scan Handlers
# =============================================================================


async def handle_scan_secrets(
    repo_path: str,
    repo_id: Optional[str] = None,
    branch: Optional[str] = None,
    include_history: bool = False,
    history_depth: int = 100,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> HandlerResult:
    """
    Trigger a secrets scan for a repository.

    POST /api/v1/codebase/{repo}/scan/secrets
    {
        "repo_path": "/path/to/repo",
        "branch": "main",
        "include_history": true,
        "history_depth": 100
    }
    """
    try:
        repo_id = repo_id or f"repo_{uuid.uuid4().hex[:12]}"
        scan_id = f"secrets_{uuid.uuid4().hex[:12]}"

        # Check if scan already running
        if repo_id in _running_secrets_scans:
            task = _running_secrets_scans[repo_id]
            if not task.done():
                return error_response("Secrets scan already in progress", 409)

        # Create initial scan result
        scan_result = SecretsScanResult(
            scan_id=scan_id,
            repository=repo_id,
            branch=branch,
            status="running",
        )

        repo_scans = _get_or_create_secrets_scans(repo_id)
        repo_scans[scan_id] = scan_result

        # Start async scan
        async def run_secrets_scan():
            try:
                scanner = SecretsScanner()

                # Scan current files
                result = await scanner.scan_repository(
                    repo_path=repo_path,
                    branch=branch,
                )

                # Optionally scan git history
                if include_history:
                    history_result = await scanner.scan_git_history(
                        repo_path=repo_path,
                        depth=history_depth,
                        branch=branch,
                    )
                    result.secrets.extend(history_result.secrets)
                    result.scanned_history = True
                    result.history_depth = history_depth

                # Update stored result
                with _secrets_scan_lock:
                    result.scan_id = scan_id
                    repo_scans[scan_id] = result

                logger.info(
                    f"[Security] Completed secrets scan {scan_id} for {repo_id}: "
                    f"{len(result.secrets)} secrets found"
                )

                # Emit security events for findings (triggers debate for critical secrets)
                await _emit_secrets_events(result, repo_id, scan_id, workspace_id)

            except Exception as e:
                logger.exception(f"Secrets scan {scan_id} failed: {e}")
                with _secrets_scan_lock:
                    scan_result.status = "failed"
                    scan_result.error = str(e)
                    scan_result.completed_at = datetime.now(timezone.utc)

            finally:
                if repo_id in _running_secrets_scans:
                    del _running_secrets_scans[repo_id]

        # Create and store task
        task = asyncio.create_task(run_secrets_scan())
        _running_secrets_scans[repo_id] = task

        logger.info(f"[Security] Started secrets scan {scan_id} for {repo_id}")

        return success_response({
            "scan_id": scan_id,
            "status": "running",
            "repository": repo_id,
            "include_history": include_history,
        })

    except Exception as e:
        logger.exception(f"Failed to start secrets scan: {e}")
        return error_response(str(e), 500)


async def handle_get_secrets_scan_status(
    repo_id: str,
    scan_id: Optional[str] = None,
) -> HandlerResult:
    """
    Get secrets scan status/result.

    GET /api/v1/codebase/{repo}/scan/secrets/latest
    GET /api/v1/codebase/{repo}/scan/secrets/{scan_id}
    """
    try:
        repo_scans = _get_or_create_secrets_scans(repo_id)

        if scan_id:
            # Get specific scan
            scan = repo_scans.get(scan_id)
            if not scan:
                return error_response("Secrets scan not found", 404)
            return success_response({"scan_result": scan.to_dict()})
        else:
            # Get latest scan
            if not repo_scans:
                return error_response("No secrets scans found for repository", 404)

            # Sort by start time and get latest
            latest = max(repo_scans.values(), key=lambda s: s.started_at)
            return success_response({"scan_result": latest.to_dict()})

    except Exception as e:
        logger.exception(f"Failed to get secrets scan status: {e}")
        return error_response(str(e), 500)


async def handle_get_secrets(
    repo_id: str,
    severity: Optional[str] = None,
    secret_type: Optional[str] = None,
    include_history: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> HandlerResult:
    """
    Get secrets from latest scan.

    GET /api/v1/codebase/{repo}/secrets
    Query params: severity, secret_type, include_history, limit, offset
    """
    try:
        repo_scans = _get_or_create_secrets_scans(repo_id)

        if not repo_scans:
            return error_response("No secrets scans found for repository", 404)

        # Get latest completed scan
        completed_scans = [s for s in repo_scans.values() if s.status == "completed"]
        if not completed_scans:
            return error_response("No completed secrets scans found", 404)

        latest = max(completed_scans, key=lambda s: s.started_at)

        # Get secrets
        secrets = [s.to_dict() for s in latest.secrets]

        # Filter
        if severity:
            secrets = [s for s in secrets if s["severity"] == severity]
        if secret_type:
            secrets = [s for s in secrets if s["secret_type"] == secret_type]
        if not include_history:
            secrets = [s for s in secrets if not s["is_in_history"]]

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}
        secrets.sort(key=lambda s: severity_order.get(s["severity"], 5))

        # Paginate
        total = len(secrets)
        secrets = secrets[offset : offset + limit]

        return success_response({
            "secrets": secrets,
            "total": total,
            "limit": limit,
            "offset": offset,
            "scan_id": latest.scan_id,
        })

    except Exception as e:
        logger.exception(f"Failed to get secrets: {e}")
        return error_response(str(e), 500)


async def handle_list_secrets_scans(
    repo_id: str,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> HandlerResult:
    """
    List secrets scan history for a repository.

    GET /api/v1/codebase/{repo}/scans/secrets
    """
    try:
        repo_scans = _get_or_create_secrets_scans(repo_id)

        scans = list(repo_scans.values())

        # Filter by status
        if status:
            scans = [s for s in scans if s.status == status]

        # Sort by start time descending
        scans.sort(key=lambda s: s.started_at, reverse=True)

        # Paginate
        total = len(scans)
        scans = scans[offset : offset + limit]

        return success_response({
            "scans": [
                {
                    "scan_id": s.scan_id,
                    "status": s.status,
                    "started_at": s.started_at.isoformat(),
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                    "files_scanned": s.files_scanned,
                    "scanned_history": s.scanned_history,
                    "history_depth": s.history_depth,
                    "summary": {
                        "total_secrets": len(s.secrets),
                        "critical_count": s.critical_count,
                        "high_count": s.high_count,
                        "medium_count": s.medium_count,
                        "low_count": s.low_count,
                    }
                    if s.status == "completed"
                    else None,
                }
                for s in scans
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        })

    except Exception as e:
        logger.exception(f"Failed to list secrets scans: {e}")
        return error_response(str(e), 500)


# =============================================================================
# Handler Class
# =============================================================================


class SecurityHandler(BaseHandler):
    """
    HTTP handler for codebase security endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/cve",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/codebase/",
        "/api/v1/cve/",
    ]

    def __init__(self, ctx: Dict[str, Any]):
        """Initialize with server context."""
        super().__init__(ctx)  # type: ignore[arg-type]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                # Check for security-related paths
                if (
                    "/scan" in path
                    or "/vulnerabilities" in path
                    or "/cve/" in path
                    or "/secrets" in path
                    or "/sbom" in path
                ):
                    return True
        return False

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route security endpoint requests."""
        return None

    async def handle_post_scan(self, data: Dict[str, Any], repo_id: str) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/scan"""
        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path required", 400)

        return await handle_scan_repository(
            repo_path=repo_path,
            repo_id=repo_id,
            branch=data.get("branch"),
            commit_sha=data.get("commit_sha"),
            workspace_id=data.get("workspace_id"),
            user_id=self._get_user_id(),
        )

    async def handle_get_scan_latest(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/latest"""
        return await handle_get_scan_status(repo_id=repo_id)

    async def handle_get_scan(
        self, params: Dict[str, Any], repo_id: str, scan_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/{scan_id}"""
        return await handle_get_scan_status(repo_id=repo_id, scan_id=scan_id)

    async def handle_get_vulnerabilities(
        self, params: Dict[str, Any], repo_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/vulnerabilities"""
        return await handle_get_vulnerabilities(
            repo_id=repo_id,
            severity=params.get("severity"),
            package=params.get("package"),
            ecosystem=params.get("ecosystem"),
            limit=int(params.get("limit", 100)),
            offset=int(params.get("offset", 0)),
        )

    async def handle_get_cve(self, params: Dict[str, Any], cve_id: str) -> HandlerResult:
        """GET /api/v1/cve/{cve_id}"""
        return await handle_get_cve_details(cve_id=cve_id)

    async def handle_list_scans(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scans"""
        return await handle_list_scans(
            repo_id=repo_id,
            status=params.get("status"),
            limit=int(params.get("limit", 20)),
            offset=int(params.get("offset", 0)),
        )

    # =========================================================================
    # Secrets Scan Endpoints
    # =========================================================================

    async def handle_post_secrets_scan(self, data: Dict[str, Any], repo_id: str) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/scan/secrets"""
        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path required", 400)

        return await handle_scan_secrets(
            repo_path=repo_path,
            repo_id=repo_id,
            branch=data.get("branch"),
            include_history=data.get("include_history", False),
            history_depth=int(data.get("history_depth", 100)),
            workspace_id=data.get("workspace_id"),
            user_id=self._get_user_id(),
        )

    async def handle_get_secrets_scan_latest(
        self, params: Dict[str, Any], repo_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/secrets/latest"""
        return await handle_get_secrets_scan_status(repo_id=repo_id)

    async def handle_get_secrets_scan(
        self, params: Dict[str, Any], repo_id: str, scan_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/secrets/{scan_id}"""
        return await handle_get_secrets_scan_status(repo_id=repo_id, scan_id=scan_id)

    async def handle_get_secrets(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/secrets"""
        return await handle_get_secrets(
            repo_id=repo_id,
            severity=params.get("severity"),
            secret_type=params.get("secret_type"),
            include_history=params.get("include_history", "true").lower() == "true",
            limit=int(params.get("limit", 100)),
            offset=int(params.get("offset", 0)),
        )

    async def handle_list_secrets_scans(
        self, params: Dict[str, Any], repo_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scans/secrets"""
        return await handle_list_secrets_scans(
            repo_id=repo_id,
            status=params.get("status"),
            limit=int(params.get("limit", 20)),
            offset=int(params.get("offset", 0)),
        )

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"

    async def handle_scan_sast(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/scan/sast"""
        return await handle_scan_sast(
            repo_path=params.get("repo_path", ""),
            repo_id=repo_id,
            rule_sets=params.get("rule_sets"),
            workspace_id=params.get("workspace_id"),
        )

    async def handle_get_sast_scan_status(
        self, params: Dict[str, Any], repo_id: str, scan_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/sast/{scan_id}"""
        return await handle_get_sast_scan_status(repo_id=repo_id, scan_id=scan_id)

    async def handle_get_sast_findings(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sast/findings"""
        return await handle_get_sast_findings(
            repo_id=repo_id,
            severity=params.get("severity"),
            owasp_category=params.get("owasp_category"),
            limit=int(params.get("limit", 100)),
            offset=int(params.get("offset", 0)),
        )

    async def handle_get_owasp_summary(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sast/owasp-summary"""
        return await handle_get_owasp_summary(repo_id=repo_id)

    # =========================================================================
    # SBOM Handler Methods
    # =========================================================================

    async def handle_post_sbom(self, data: Dict[str, Any], repo_id: str) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/sbom - Generate SBOM"""
        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path required", 400)

        return await handle_generate_sbom(
            repo_path=repo_path,
            repo_id=repo_id,
            format=data.get("format", "cyclonedx-json"),
            workspace_id=data.get("workspace_id"),
        )

    async def handle_get_sbom_latest(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sbom/latest"""
        return await handle_get_sbom(repo_id=repo_id)

    async def handle_get_sbom_by_id(
        self, params: Dict[str, Any], repo_id: str, sbom_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sbom/{sbom_id}"""
        return await handle_get_sbom(repo_id=repo_id, sbom_id=sbom_id)

    async def handle_list_sbom(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sbom/list"""
        return await handle_list_sboms(repo_id=repo_id)

    async def handle_download_sbom_content(
        self, params: Dict[str, Any], repo_id: str, sbom_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/sbom/{sbom_id}/download"""
        return await handle_download_sbom(repo_id=repo_id, sbom_id=sbom_id)

    async def handle_compare_sbom(
        self, data: Dict[str, Any], repo_id: str
    ) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/sbom/compare"""
        sbom_id_a = data.get("sbom_id_a")
        sbom_id_b = data.get("sbom_id_b")

        if not sbom_id_a or not sbom_id_b:
            return error_response("sbom_id_a and sbom_id_b required", 400)

        return await handle_compare_sboms(
            repo_id=repo_id,
            sbom_id_a=sbom_id_a,
            sbom_id_b=sbom_id_b,
        )


# =============================================================================
# SAST Scan Handlers
# =============================================================================


async def handle_scan_sast(
    repo_path: str,
    repo_id: Optional[str] = None,
    rule_sets: Optional[list] = None,
    workspace_id: Optional[str] = None,
) -> HandlerResult:
    """
    Trigger a SAST scan for a repository.

    POST /api/v1/codebase/{repo}/scan/sast
    {
        "repo_path": "/path/to/repo",
        "rule_sets": ["p/owasp-top-ten", "p/security-audit"]
    }
    """
    try:
        repo_id = repo_id or f"repo_{uuid.uuid4().hex[:12]}"
        scan_id = f"sast_{uuid.uuid4().hex[:12]}"

        # Check if scan already running
        if repo_id in _running_sast_scans:
            task = _running_sast_scans[repo_id]
            if not task.done():
                return error_response("SAST scan already in progress", 409)

        # Get or create storage
        with _sast_scan_lock:
            if repo_id not in _sast_scan_results:
                _sast_scan_results[repo_id] = {}

        # Start async scan
        async def run_sast_scan():
            try:
                scanner = _get_sast_scanner()
                await scanner.initialize()

                result = await scanner.scan_repository(
                    repo_path=repo_path,
                    rule_sets=rule_sets,
                    scan_id=scan_id,
                )

                # Store result
                with _sast_scan_lock:
                    _sast_scan_results[repo_id][scan_id] = result

                logger.info(
                    f"[SAST] Completed scan {scan_id} for {repo_id}: "
                    f"{len(result.findings)} findings"
                )

                # Emit security events for critical/high findings
                critical_findings = [
                    f for f in result.findings if f.severity.value in ("critical", "error")
                ]
                if critical_findings:
                    await _emit_sast_events(result, repo_id, scan_id, workspace_id)

            except Exception as e:
                logger.exception(f"[SAST] Scan failed for {repo_id}: {e}")
            finally:
                if repo_id in _running_sast_scans:
                    del _running_sast_scans[repo_id]

        task = asyncio.create_task(run_sast_scan())
        _running_sast_scans[repo_id] = task

        return success_response({
            "message": "SAST scan started",
            "scan_id": scan_id,
            "repo_id": repo_id,
        })

    except Exception as e:
        logger.exception(f"[SAST] Failed to start scan: {e}")
        return error_response(str(e), 500)


async def handle_get_sast_scan_status(
    repo_id: str,
    scan_id: str,
) -> HandlerResult:
    """Get status and results of a SAST scan."""
    try:
        with _sast_scan_lock:
            if repo_id not in _sast_scan_results:
                return error_response("Repository not found", 404)

            repo_scans = _sast_scan_results[repo_id]
            if scan_id not in repo_scans:
                # Check if still running
                if repo_id in _running_sast_scans:
                    return success_response({
                        "scan_id": scan_id,
                        "status": "running",
                        "findings_count": 0,
                    })
                return error_response("Scan not found", 404)

            result = repo_scans[scan_id]
            return success_response({
                "scan_id": scan_id,
                "status": "completed",
                **result.to_dict(),
            })

    except Exception as e:
        logger.exception(f"[SAST] Failed to get scan status: {e}")
        return error_response(str(e), 500)


async def handle_get_sast_findings(
    repo_id: str,
    severity: Optional[str] = None,
    owasp_category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> HandlerResult:
    """Get SAST findings for a repository."""
    try:
        with _sast_scan_lock:
            if repo_id not in _sast_scan_results:
                return error_response("Repository not found", 404)

            repo_scans = _sast_scan_results[repo_id]
            if not repo_scans:
                return success_response({
                    "findings": [],
                    "total": 0,
                })

            # Get latest scan
            latest_scan = max(repo_scans.values(), key=lambda s: s.scanned_at)
            findings = latest_scan.findings

            # Filter by severity
            if severity:
                findings = [f for f in findings if f.severity.value == severity]

            # Filter by OWASP category
            if owasp_category:
                findings = [f for f in findings if owasp_category in f.owasp_category.value]

            total = len(findings)
            findings = findings[offset : offset + limit]

            return success_response({
                "findings": [f.to_dict() for f in findings],
                "total": total,
                "limit": limit,
                "offset": offset,
                "scan_id": latest_scan.scan_id,
            })

    except Exception as e:
        logger.exception(f"[SAST] Failed to get findings: {e}")
        return error_response(str(e), 500)


async def handle_get_owasp_summary(repo_id: str) -> HandlerResult:
    """Get OWASP Top 10 summary for a repository."""
    try:
        with _sast_scan_lock:
            if repo_id not in _sast_scan_results:
                return error_response("Repository not found", 404)

            repo_scans = _sast_scan_results[repo_id]
            if not repo_scans:
                return success_response({
                    "owasp_summary": {},
                    "total_findings": 0,
                })

            # Get latest scan
            latest_scan = max(repo_scans.values(), key=lambda s: s.scanned_at)

            # Get OWASP summary
            scanner = _get_sast_scanner()
            summary = await scanner.get_owasp_summary(latest_scan.findings)

            return success_response({
                "scan_id": latest_scan.scan_id,
                **summary,
            })

    except Exception as e:
        logger.exception(f"[SAST] Failed to get OWASP summary: {e}")
        return error_response(str(e), 500)


async def _emit_sast_events(
    result: SASTScanResult,
    repo_id: str,
    scan_id: str,
    workspace_id: Optional[str] = None,
) -> None:
    """Emit security events for SAST findings."""
    try:
        emitter = get_security_emitter()

        for finding in result.findings:
            if finding.severity.value not in ("critical", "error"):
                continue

            severity = (
                SecuritySeverity.CRITICAL
                if finding.severity.value == "critical"
                else SecuritySeverity.HIGH
            )

            sec_finding = SecurityFinding(
                finding_id=f"{scan_id}:{finding.rule_id}:{finding.line_start}",
                title=finding.message[:100],
                description=finding.message,
                severity=severity,
                category=finding.owasp_category.value,
                source="sast_scanner",
                location=f"{finding.file_path}:{finding.line_start}",
                cwe_id=finding.cwe_ids[0] if finding.cwe_ids else None,
                remediation=finding.remediation or "Review and fix the security issue",
                confidence=finding.confidence,
            )

            event = SecurityEvent(
                event_type=SecurityEventType.SAST_FINDING,
                severity=severity,
                source="sast_scanner",
                findings=[sec_finding],
                scan_id=scan_id,
                repository_id=repo_id,
                metadata={
                    "rule_id": finding.rule_id,
                    "language": finding.language,
                    "owasp_category": finding.owasp_category.value,
                },
            )

            await emitter.emit(event)

    except Exception as e:
        logger.warning(f"Failed to emit SAST events: {e}")


# =============================================================================
# SBOM Generation Handlers
# =============================================================================

# SBOM storage
_sbom_results: Dict[str, Dict[str, SBOMResult]] = {}  # repo_id -> {sbom_id -> result}
_sbom_lock = threading.Lock()
_running_sbom_generations: Dict[str, asyncio.Task] = {}


def _get_or_create_sbom_results(repo_id: str) -> Dict[str, SBOMResult]:
    """Get or create SBOM storage for a repository."""
    with _sbom_lock:
        if repo_id not in _sbom_results:
            _sbom_results[repo_id] = {}
        return _sbom_results[repo_id]


def _get_sbom_generator() -> SBOMGenerator:
    """Get or create SBOMGenerator from service registry."""
    registry = ServiceRegistry.get()
    if not registry.has(SBOMGenerator):
        generator = SBOMGenerator()
        registry.register(SBOMGenerator, generator)
        logger.info("Registered SBOMGenerator with service registry")
    return registry.resolve(SBOMGenerator)


async def handle_generate_sbom(
    repo_path: str,
    repo_id: Optional[str] = None,
    format: str = "cyclonedx-json",
    project_name: Optional[str] = None,
    project_version: Optional[str] = None,
    include_dev: bool = True,
    include_vulnerabilities: bool = True,
    branch: Optional[str] = None,
    commit_sha: Optional[str] = None,
) -> HandlerResult:
    """
    Generate SBOM for a repository.

    POST /api/v1/codebase/{repo}/sbom
    {
        "repo_path": "/path/to/repo",
        "format": "cyclonedx-json",  // cyclonedx-json, cyclonedx-xml, spdx-json, spdx-tv
        "project_name": "MyProject",
        "project_version": "1.0.0",
        "include_dev": true,
        "include_vulnerabilities": true
    }

    Returns:
        SBOM generation result with content and metadata
    """
    try:
        repo_id = repo_id or f"repo_{uuid.uuid4().hex[:12]}"
        sbom_id = f"sbom_{uuid.uuid4().hex[:12]}"

        # Check if generation already running
        if repo_id in _running_sbom_generations:
            task = _running_sbom_generations[repo_id]
            if not task.done():
                return error_response("SBOM generation already in progress", 409)

        # Parse format
        try:
            sbom_format = SBOMFormat(format)
        except ValueError:
            return error_response(
                f"Invalid format: {format}. Valid formats: "
                "cyclonedx-json, cyclonedx-xml, spdx-json, spdx-tv",
                400,
            )

        # Run generation
        generator = _get_sbom_generator()
        generator.include_dev_dependencies = include_dev
        generator.include_vulnerabilities = include_vulnerabilities

        result = await generator.generate_from_repo(
            repo_path=repo_path,
            format=sbom_format,
            project_name=project_name,
            project_version=project_version,
            branch=branch,
            commit_sha=commit_sha,
        )

        # Store result
        repo_results = _get_or_create_sbom_results(repo_id)
        with _sbom_lock:
            repo_results[sbom_id] = result

        logger.info(
            f"[SBOM] Generated {format} for {repo_id}: "
            f"{result.component_count} components, {result.vulnerability_count} vulnerabilities"
        )

        return success_response({
            "sbom_id": sbom_id,
            "repository": repo_id,
            "format": result.format.value,
            "filename": result.filename,
            "component_count": result.component_count,
            "vulnerability_count": result.vulnerability_count,
            "license_count": result.license_count,
            "generated_at": result.generated_at.isoformat(),
            "content": result.content,
            "errors": result.errors,
        })

    except Exception as e:
        logger.exception(f"Failed to generate SBOM: {e}")
        return error_response(str(e), 500)


async def handle_get_sbom(
    repo_id: str,
    sbom_id: Optional[str] = None,
) -> HandlerResult:
    """
    Get SBOM content.

    GET /api/v1/codebase/{repo}/sbom/latest
    GET /api/v1/codebase/{repo}/sbom/{sbom_id}
    """
    try:
        repo_results = _get_or_create_sbom_results(repo_id)

        if sbom_id:
            # Get specific SBOM
            result = repo_results.get(sbom_id)
            if not result:
                return error_response(f"SBOM not found: {sbom_id}", 404)
        else:
            # Get latest SBOM
            if not repo_results:
                return error_response("No SBOMs generated for this repository", 404)
            result = max(repo_results.values(), key=lambda r: r.generated_at)

        return success_response({
            "sbom_id": sbom_id or "sbom_latest",
            "repository": repo_id,
            "format": result.format.value,
            "filename": result.filename,
            "component_count": result.component_count,
            "vulnerability_count": result.vulnerability_count,
            "license_count": result.license_count,
            "generated_at": result.generated_at.isoformat(),
            "content": result.content,
            "errors": result.errors,
        })

    except Exception as e:
        logger.exception(f"Failed to get SBOM: {e}")
        return error_response(str(e), 500)


async def handle_list_sboms(
    repo_id: str,
    limit: int = 10,
) -> HandlerResult:
    """
    List SBOMs for a repository.

    GET /api/v1/codebase/{repo}/sbom/list
    """
    try:
        repo_results = _get_or_create_sbom_results(repo_id)

        # Sort by generated_at descending
        sorted_results = sorted(
            repo_results.items(),
            key=lambda x: x[1].generated_at,
            reverse=True,
        )[:limit]

        sboms = [
            {
                "sbom_id": sbom_id,
                "format": result.format.value,
                "filename": result.filename,
                "component_count": result.component_count,
                "vulnerability_count": result.vulnerability_count,
                "license_count": result.license_count,
                "generated_at": result.generated_at.isoformat(),
            }
            for sbom_id, result in sorted_results
        ]

        return success_response({
            "repository": repo_id,
            "count": len(sboms),
            "sboms": sboms,
        })

    except Exception as e:
        logger.exception(f"Failed to list SBOMs: {e}")
        return error_response(str(e), 500)


async def handle_download_sbom(
    repo_id: str,
    sbom_id: str,
) -> HandlerResult:
    """
    Download SBOM content (raw).

    GET /api/v1/codebase/{repo}/sbom/{sbom_id}/download

    Returns the raw SBOM content with appropriate content-type header info.
    """
    try:
        repo_results = _get_or_create_sbom_results(repo_id)
        result = repo_results.get(sbom_id)

        if not result:
            return error_response(f"SBOM not found: {sbom_id}", 404)

        # Determine content type
        content_types = {
            SBOMFormat.CYCLONEDX_JSON: "application/json",
            SBOMFormat.CYCLONEDX_XML: "application/xml",
            SBOMFormat.SPDX_JSON: "application/json",
            SBOMFormat.SPDX_TV: "text/plain",
        }
        content_type = content_types.get(result.format, "application/octet-stream")

        return success_response({
            "content": result.content,
            "filename": result.filename,
            "content_type": content_type,
        })

    except Exception as e:
        logger.exception(f"Failed to download SBOM: {e}")
        return error_response(str(e), 500)


async def handle_compare_sboms(
    repo_id: str,
    sbom_id_a: str,
    sbom_id_b: str,
) -> HandlerResult:
    """
    Compare two SBOMs to find differences.

    POST /api/v1/codebase/{repo}/sbom/compare
    {
        "sbom_id_a": "sbom_abc123",
        "sbom_id_b": "sbom_def456"
    }
    """
    try:
        repo_results = _get_or_create_sbom_results(repo_id)

        result_a = repo_results.get(sbom_id_a)
        result_b = repo_results.get(sbom_id_b)

        if not result_a:
            return error_response(f"SBOM not found: {sbom_id_a}", 404)
        if not result_b:
            return error_response(f"SBOM not found: {sbom_id_b}", 404)

        # Parse components from both (simplified - works for JSON formats)
        import json

        def extract_components(content: str, format: SBOMFormat) -> Dict[str, str]:
            """Extract component name -> version mapping."""
            components = {}
            try:
                if format in (SBOMFormat.CYCLONEDX_JSON, SBOMFormat.SPDX_JSON):
                    data = json.loads(content)
                    if format == SBOMFormat.CYCLONEDX_JSON:
                        for comp in data.get("components", []):
                            name = comp.get("name", "")
                            if comp.get("group"):
                                name = f"{comp['group']}/{name}"
                            components[name] = comp.get("version", "")
                    else:  # SPDX
                        for pkg in data.get("packages", []):
                            components[pkg.get("name", "")] = pkg.get("versionInfo", "")
            except Exception:
                pass
            return components

        components_a = extract_components(result_a.content, result_a.format)
        components_b = extract_components(result_b.content, result_b.format)

        all_names = set(components_a.keys()) | set(components_b.keys())

        added = []
        removed = []
        updated = []
        unchanged = []

        for name in sorted(all_names):
            v_a = components_a.get(name)
            v_b = components_b.get(name)

            if v_a and not v_b:
                removed.append({"name": name, "version": v_a})
            elif v_b and not v_a:
                added.append({"name": name, "version": v_b})
            elif v_a != v_b:
                updated.append({"name": name, "old_version": v_a, "new_version": v_b})
            else:
                unchanged.append({"name": name, "version": v_a})

        return success_response({
            "sbom_a": {
                "sbom_id": sbom_id_a,
                "generated_at": result_a.generated_at.isoformat(),
                "component_count": result_a.component_count,
            },
            "sbom_b": {
                "sbom_id": sbom_id_b,
                "generated_at": result_b.generated_at.isoformat(),
                "component_count": result_b.component_count,
            },
            "diff": {
                "added": added,
                "removed": removed,
                "updated": updated,
                "unchanged_count": len(unchanged),
            },
            "summary": {
                "total_added": len(added),
                "total_removed": len(removed),
                "total_updated": len(updated),
                "total_unchanged": len(unchanged),
            },
        })

    except Exception as e:
        logger.exception(f"Failed to compare SBOMs: {e}")
        return error_response(str(e), 500)
