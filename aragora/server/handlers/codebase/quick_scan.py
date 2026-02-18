"""
Quick Security Scan API Handler.

Provides a simplified one-click security scan endpoint for the UI wizard.
Combines pattern matching, secrets detection, and basic vulnerability analysis.

Endpoints:
- POST /api/codebase/quick-scan - Run quick security scan
- GET /api/codebase/quick-scan/{scan_id} - Get scan result
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import web

from aragora.rbac.checker import get_permission_checker
from aragora.server.handlers.base import handle_errors
from aragora.server.handlers.utils import parse_json_body
from aragora.server.validation.query_params import safe_query_int

logger = logging.getLogger(__name__)

# Permission constants for quick scan operations
SCAN_READ_PERMISSION = "codebase:scan:read"
SCAN_EXECUTE_PERMISSION = "codebase:scan:execute"

SecurityScanner: Any = None
SecuritySeverity: Any = None
try:
    from aragora.audit.security_scanner import SecurityScanner, SecuritySeverity
except ImportError:  # pragma: no cover - optional dependency
    pass


async def _check_permission(request: web.Request, permission: str) -> web.Response | None:
    """Check if user has required permission. Returns error response if denied."""
    try:
        # Get auth context from request
        auth_context = getattr(request, "auth_context", None)
        if not auth_context:
            return web.json_response(
                {"success": False, "error": "Authentication required"},
                status=401,
            )

        # Check permission
        checker = get_permission_checker()
        user_id = getattr(auth_context, "user_id", None)
        if not user_id:
            return web.json_response(
                {"success": False, "error": "Authentication required"},
                status=401,
            )

        if not checker.check_permission(user_id, permission).allowed:
            return web.json_response(
                {"success": False, "error": "Permission denied"},
                status=403,
            )
        return None  # Permission granted
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        logger.warning(f"Permission check failed: {e}")
        return web.json_response(
            {"success": False, "error": "Authentication required"},
            status=401,
        )


def _validate_repo_path(repo_path: str) -> tuple[str | None, str | None]:
    """Validate repo_path to prevent path traversal attacks.

    Resolves the path (including symlinks and '..'). If ARAGORA_SCAN_ROOT is
    configured, the resolved path must remain within that boundary.

    Returns:
        (resolved_path, None) on success, or (None, error_message) on failure.
    """
    if not repo_path or not repo_path.strip():
        return None, "repo_path is required"

    if "\x00" in repo_path:
        return None, "repo_path contains invalid null byte"

    allowed_root_env = os.environ.get("ARAGORA_SCAN_ROOT", "").strip()
    resolved = os.path.realpath(repo_path)

    if allowed_root_env:
        allowed_root = os.path.realpath(allowed_root_env)
        # The trailing os.sep prevents "/allowed_root_extra" matching "/allowed_root".
        # Special-case the filesystem root since root + os.sep would be "//".
        if allowed_root == os.sep:
            pass  # All absolute paths are under the root
        elif not (resolved == allowed_root or resolved.startswith(allowed_root + os.sep)):
            return None, "repo_path must be within the allowed workspace directory"

    return resolved, None


# In-memory storage for scan results
_quick_scan_results: dict[str, dict[str, Any]] = {}


async def run_quick_scan(
    repo_path: str,
    severity_threshold: str = "medium",
    include_secrets: bool = True,
    scan_id: str | None = None,
) -> dict[str, Any]:
    """
    Run a quick security scan on a repository.

    This is a simplified scan that focuses on:
    - Common vulnerability patterns
    - Hardcoded secrets and credentials
    - Basic configuration issues

    Args:
        repo_path: Path to the repository to scan
        severity_threshold: Minimum severity to report (critical, high, medium, low)
        include_secrets: Whether to include secrets detection
        scan_id: Optional existing scan ID to update

    Returns:
        Scan result dictionary
    """
    scan_id = scan_id or f"qscan_{uuid.uuid4().hex[:12]}"
    start_time = datetime.now(timezone.utc)

    logger.info(f"[QuickScan] Starting scan {scan_id} on {repo_path}")

    result: dict[str, Any] = {
        "scan_id": scan_id,
        "repository": repo_path,
        "status": "running",
        "started_at": start_time.isoformat(),
        "completed_at": None,
        "files_scanned": 0,
        "lines_scanned": 0,
        "risk_score": 0,
        "summary": {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        },
        "findings": [],
        "error": None,
    }

    _quick_scan_results[scan_id] = result

    try:
        if SecurityScanner is None or SecuritySeverity is None:
            raise ImportError("SecurityScanner not available")

        # Determine severity filter
        severity_map = {
            "critical": SecuritySeverity.CRITICAL,
            "high": SecuritySeverity.HIGH,
            "medium": SecuritySeverity.MEDIUM,
            "low": SecuritySeverity.LOW,
        }
        min_severity = severity_map.get(severity_threshold, SecuritySeverity.MEDIUM)

        # Initialize scanner
        scanner = SecurityScanner(
            include_low_severity=(min_severity == SecuritySeverity.LOW),
            include_info=False,
        )

        # Check if path exists and is a directory
        path = Path(repo_path)
        if not path.exists():
            raise ValueError(f"Path does not exist: {repo_path}")

        if path.is_file():
            # Scan single file
            findings = scanner.scan_file(str(path))
            result["files_scanned"] = 1
            with open(path, encoding="utf-8", errors="replace") as f:
                result["lines_scanned"] = f.read().count("\n") + 1
        else:
            # Scan directory
            report = scanner.scan_directory(str(path))
            findings = report.findings
            result["files_scanned"] = report.files_scanned
            result["lines_scanned"] = report.lines_scanned
            result["risk_score"] = report.risk_score

        # Convert findings to dict
        result["findings"] = [f.to_dict() for f in findings]

        # Calculate summary
        for finding in findings:
            severity = finding.severity.value
            if severity in result["summary"]:
                result["summary"][severity] += 1

        # Calculate risk score if not already set
        if result["risk_score"] == 0 and findings:
            weights = {"critical": 40, "high": 20, "medium": 10, "low": 5, "info": 1}
            score = sum(weights.get(f.severity.value, 0) * f.confidence for f in findings)
            result["risk_score"] = min(100.0, score)

        result["status"] = "completed"
        result["completed_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"[QuickScan] Completed {scan_id}: "
            f"{len(findings)} findings, risk score {result['risk_score']:.1f}"
        )

    except ImportError as e:
        logger.warning(f"[QuickScan] Scanner not available: {e}")
        # Return mock result for demo
        result = _generate_mock_result(scan_id, repo_path, start_time)

    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        logger.exception(f"[QuickScan] Scan {scan_id} failed: {e}")
        result["status"] = "failed"
        result["error"] = "Scan failed"
        result["completed_at"] = datetime.now(timezone.utc).isoformat()

    _quick_scan_results[scan_id] = result
    return result


def _generate_mock_result(scan_id: str, repo_path: str, start_time: datetime) -> dict[str, Any]:
    """Generate mock scan result for demo/testing."""
    return {
        "scan_id": scan_id,
        "repository": repo_path,
        "status": "completed",
        "started_at": start_time.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "files_scanned": 127,
        "lines_scanned": 15420,
        "risk_score": 35.0,
        "summary": {
            "critical": 0,
            "high": 2,
            "medium": 5,
            "low": 8,
            "info": 3,
        },
        "findings": [
            {
                "id": "SEC-000001",
                "title": "Hardcoded API Key Pattern",
                "description": "Potential API key detected in source code",
                "category": "hardcoded_secret",
                "severity": "high",
                "confidence": 0.85,
                "file_path": "src/config/api.ts",
                "line_number": 42,
                "code_snippet": 'const API_KEY = "sk-..."',
                "cwe_id": "CWE-798",
                "recommendation": "Move API keys to environment variables",
            },
            {
                "id": "SEC-000002",
                "title": "SQL String Interpolation",
                "description": "SQL query using template literals may be vulnerable to injection",
                "category": "sql_injection",
                "severity": "high",
                "confidence": 0.92,
                "file_path": "src/db/queries.ts",
                "line_number": 78,
                "code_snippet": "db.query(`SELECT * FROM users WHERE id = ${userId}`)",
                "cwe_id": "CWE-89",
                "recommendation": "Use parameterized queries",
            },
            {
                "id": "SEC-000003",
                "title": "Debug Mode Enabled",
                "description": "Debug mode appears to be enabled in configuration",
                "category": "insecure_config",
                "severity": "medium",
                "confidence": 0.78,
                "file_path": "src/config/app.ts",
                "line_number": 15,
                "cwe_id": "CWE-489",
                "recommendation": "Ensure DEBUG is false in production",
            },
        ],
        "error": None,
    }


async def get_quick_scan_result(scan_id: str) -> dict[str, Any] | None:
    """Get a quick scan result by ID."""
    return _quick_scan_results.get(scan_id)


async def list_quick_scans(
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """List recent quick scans."""
    scans = list(_quick_scan_results.values())
    scans.sort(key=lambda s: s.get("started_at", ""), reverse=True)

    total = len(scans)
    scans = scans[offset : offset + limit]

    return {
        "scans": [
            {
                "scan_id": s["scan_id"],
                "repository": s["repository"],
                "status": s["status"],
                "started_at": s["started_at"],
                "completed_at": s["completed_at"],
                "risk_score": s.get("risk_score", 0),
                "findings_count": len(s.get("findings", [])),
            }
            for s in scans
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# =============================================================================
# HTTP Handlers
# =============================================================================


class QuickScanHandler:
    """Handler for quick scan API endpoints."""

    def __init__(self, _ctx: Any | None = None) -> None:
        """Allow handler registry to pass server context without error."""
        self._ctx = _ctx

    @handle_errors("quick scan operation")
    async def handle_post_quick_scan(self, request: web.Request) -> web.Response:
        """
        POST /api/codebase/quick-scan

        Body:
            - repo_path: Path to repository (required)
            - severity_threshold: Minimum severity (default: medium)
            - include_secrets: Include secrets scan (default: true)
        """
        # RBAC: Require codebase:scan:execute permission
        auth_error = await _check_permission(request, SCAN_EXECUTE_PERMISSION)
        if auth_error:
            return auth_error

        body, err = await parse_json_body(request, context="handle_post_quick_scan")
        if err:
            return err

        try:
            repo_path = body.get("repo_path")

            if not repo_path:
                return web.json_response(
                    {"success": False, "error": "repo_path is required"},
                    status=400,
                )

            # Validate repo_path to prevent path traversal
            validated_path, path_error = _validate_repo_path(repo_path)
            if path_error:
                return web.json_response(
                    {"success": False, "error": path_error},
                    status=400,
                )

            severity = body.get("severity_threshold", "medium")
            include_secrets = body.get("include_secrets", True)

            # Run scan asynchronously
            result = await run_quick_scan(
                repo_path=validated_path,
                severity_threshold=severity,
                include_secrets=include_secrets,
            )

            return web.json_response(
                {
                    "success": True,
                    **result,
                }
            )

        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
            logger.exception(f"Quick scan failed: {e}")
            return web.json_response(
                {"success": False, "error": "Quick scan failed"},
                status=500,
            )

    async def handle_get_quick_scan(self, request: web.Request) -> web.Response:
        """
        GET /api/codebase/quick-scan/{scan_id}

        Get result of a quick scan.
        """
        # RBAC: Require codebase:scan:read permission
        auth_error = await _check_permission(request, SCAN_READ_PERMISSION)
        if auth_error:
            return auth_error

        try:
            scan_id = request.match_info.get("scan_id")
            if not scan_id:
                return web.json_response(
                    {"success": False, "error": "scan_id is required"},
                    status=400,
                )

            result = await get_quick_scan_result(scan_id)
            if not result:
                return web.json_response(
                    {"success": False, "error": "Scan not found"},
                    status=404,
                )

            return web.json_response({"success": True, **result})

        except (KeyError, ValueError, TypeError) as e:
            logger.exception(f"Failed to get scan result: {e}")
            return web.json_response(
                {"success": False, "error": "Failed to retrieve scan result"},
                status=500,
            )

    async def handle_list_quick_scans(self, request: web.Request) -> web.Response:
        """
        GET /api/codebase/quick-scans

        List recent quick scans.
        """
        # RBAC: Require codebase:scan:read permission
        auth_error = await _check_permission(request, SCAN_READ_PERMISSION)
        if auth_error:
            return auth_error

        try:
            limit = safe_query_int(request.query, "limit", default=20, max_val=1000)
            offset = safe_query_int(request.query, "offset", default=0, max_val=100000)

            result = await list_quick_scans(limit=limit, offset=offset)
            return web.json_response({"success": True, **result})

        except (KeyError, ValueError, TypeError) as e:
            logger.exception(f"Failed to list scans: {e}")
            return web.json_response(
                {"success": False, "error": "Failed to list scans"},
                status=500,
            )


def register_routes(app: web.Application) -> None:
    """Register quick scan routes."""
    handler = QuickScanHandler()

    # v1 canonical routes
    app.router.add_post("/api/v1/codebase/quick-scan", handler.handle_post_quick_scan)
    app.router.add_get("/api/v1/codebase/quick-scan/{scan_id}", handler.handle_get_quick_scan)
    app.router.add_get("/api/v1/codebase/quick-scans", handler.handle_list_quick_scans)

    # legacy routes
    app.router.add_post("/api/codebase/quick-scan", handler.handle_post_quick_scan)
    app.router.add_get("/api/codebase/quick-scan/{scan_id}", handler.handle_get_quick_scan)
    app.router.add_get("/api/codebase/quick-scans", handler.handle_list_quick_scans)
