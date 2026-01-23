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
)
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

_scan_results: Dict[str, Dict[str, ScanResult]] = {}  # repo_id -> {scan_id -> result}
_scan_lock = threading.Lock()
_running_scans: Dict[str, asyncio.Task] = {}


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
) -> Dict[str, Any]:
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
                return {
                    "success": False,
                    "error": "Scan already in progress",
                    "scan_id": None,
                }

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
                scanner = DependencyScanner()
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

        return {
            "success": True,
            "scan_id": scan_id,
            "status": "running",
            "repository": repo_id,
        }

    except Exception as e:
        logger.exception(f"Failed to start scan: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_scan_status(
    repo_id: str,
    scan_id: Optional[str] = None,
) -> Dict[str, Any]:
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
                return {"success": False, "error": "Scan not found"}
            return {
                "success": True,
                "scan_result": scan.to_dict(),
            }
        else:
            # Get latest scan
            if not repo_scans:
                return {"success": False, "error": "No scans found for repository"}

            # Sort by start time and get latest
            latest = max(repo_scans.values(), key=lambda s: s.started_at)
            return {
                "success": True,
                "scan_result": latest.to_dict(),
            }

    except Exception as e:
        logger.exception(f"Failed to get scan status: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_vulnerabilities(
    repo_id: str,
    severity: Optional[str] = None,
    package: Optional[str] = None,
    ecosystem: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Get vulnerabilities from latest scan.

    GET /api/v1/codebase/{repo}/vulnerabilities
    Query params: severity, package, ecosystem, limit, offset
    """
    try:
        repo_scans = _get_or_create_repo_scans(repo_id)

        if not repo_scans:
            return {"success": False, "error": "No scans found for repository"}

        # Get latest completed scan
        completed_scans = [s for s in repo_scans.values() if s.status == "completed"]
        if not completed_scans:
            return {"success": False, "error": "No completed scans found"}

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

        return {
            "success": True,
            "vulnerabilities": vulnerabilities,
            "total": total,
            "limit": limit,
            "offset": offset,
            "scan_id": latest.scan_id,
        }

    except Exception as e:
        logger.exception(f"Failed to get vulnerabilities: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_cve_details(
    cve_id: str,
) -> Dict[str, Any]:
    """
    Get CVE details from vulnerability databases.

    GET /api/v1/cve/{cve_id}
    """
    try:
        client = CVEClient()
        vuln = await client.get_cve(cve_id)

        if not vuln:
            return {"success": False, "error": f"CVE {cve_id} not found"}

        return {
            "success": True,
            "vulnerability": vuln.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to get CVE details: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_query_package_vulnerabilities(
    package_name: str,
    ecosystem: str,
    version: Optional[str] = None,
) -> Dict[str, Any]:
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

        return {
            "success": True,
            "package": package_name,
            "ecosystem": ecosystem,
            "version": version,
            "vulnerabilities": [v.to_dict() for v in vulnerabilities],
            "total": len(vulnerabilities),
        }

    except Exception as e:
        logger.exception(f"Failed to query package vulnerabilities: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_list_scans(
    repo_id: str,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
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

        return {
            "success": True,
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
        }

    except Exception as e:
        logger.exception(f"Failed to list scans: {e}")
        return {
            "success": False,
            "error": str(e),
        }


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
                if "/scan" in path or "/vulnerabilities" in path or "/cve/" in path:
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

        result = await handle_scan_repository(
            repo_path=repo_path,
            repo_id=repo_id,
            branch=data.get("branch"),
            commit_sha=data.get("commit_sha"),
            workspace_id=data.get("workspace_id"),
            user_id=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_scan_latest(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/latest"""
        result = await handle_get_scan_status(repo_id=repo_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_get_scan(
        self, params: Dict[str, Any], repo_id: str, scan_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scan/{scan_id}"""
        result = await handle_get_scan_status(repo_id=repo_id, scan_id=scan_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_get_vulnerabilities(
        self, params: Dict[str, Any], repo_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/vulnerabilities"""
        result = await handle_get_vulnerabilities(
            repo_id=repo_id,
            severity=params.get("severity"),
            package=params.get("package"),
            ecosystem=params.get("ecosystem"),
            limit=int(params.get("limit", 100)),
            offset=int(params.get("offset", 0)),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_cve(self, params: Dict[str, Any], cve_id: str) -> HandlerResult:
        """GET /api/v1/cve/{cve_id}"""
        result = await handle_get_cve_details(cve_id=cve_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_list_scans(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/scans"""
        result = await handle_list_scans(
            repo_id=repo_id,
            status=params.get("status"),
            limit=int(params.get("limit", 20)),
            offset=int(params.get("offset", 0)),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"
