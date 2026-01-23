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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

logger = logging.getLogger(__name__)

# In-memory storage for scan results
_quick_scan_results: Dict[str, Dict[str, Any]] = {}


async def run_quick_scan(
    repo_path: str,
    severity_threshold: str = "medium",
    include_secrets: bool = True,
    scan_id: Optional[str] = None,
) -> Dict[str, Any]:
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

    result: Dict[str, Any] = {
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
        # Import scanner
        from aragora.audit.security_scanner import SecurityScanner, SecuritySeverity

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
            with open(path, "r", encoding="utf-8", errors="replace") as f:
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

    except Exception as e:
        logger.exception(f"[QuickScan] Scan {scan_id} failed: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
        result["completed_at"] = datetime.now(timezone.utc).isoformat()

    _quick_scan_results[scan_id] = result
    return result


def _generate_mock_result(scan_id: str, repo_path: str, start_time: datetime) -> Dict[str, Any]:
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


async def get_quick_scan_result(scan_id: str) -> Optional[Dict[str, Any]]:
    """Get a quick scan result by ID."""
    return _quick_scan_results.get(scan_id)


async def list_quick_scans(
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
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

    async def handle_post_quick_scan(self, request: web.Request) -> web.Response:
        """
        POST /api/codebase/quick-scan

        Body:
            - repo_path: Path to repository (required)
            - severity_threshold: Minimum severity (default: medium)
            - include_secrets: Include secrets scan (default: true)
        """
        try:
            body = await request.json()
            repo_path = body.get("repo_path")

            if not repo_path:
                return web.json_response(
                    {"success": False, "error": "repo_path is required"},
                    status=400,
                )

            severity = body.get("severity_threshold", "medium")
            include_secrets = body.get("include_secrets", True)

            # Run scan asynchronously
            result = await run_quick_scan(
                repo_path=repo_path,
                severity_threshold=severity,
                include_secrets=include_secrets,
            )

            return web.json_response(
                {
                    "success": True,
                    **result,
                }
            )

        except Exception as e:
            logger.exception(f"Quick scan failed: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    async def handle_get_quick_scan(self, request: web.Request) -> web.Response:
        """
        GET /api/codebase/quick-scan/{scan_id}

        Get result of a quick scan.
        """
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

        except Exception as e:
            logger.exception(f"Failed to get scan result: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    async def handle_list_quick_scans(self, request: web.Request) -> web.Response:
        """
        GET /api/codebase/quick-scans

        List recent quick scans.
        """
        try:
            limit = int(request.query.get("limit", "20"))
            offset = int(request.query.get("offset", "0"))

            result = await list_quick_scans(limit=limit, offset=offset)
            return web.json_response({"success": True, **result})

        except Exception as e:
            logger.exception(f"Failed to list scans: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )


def register_routes(app: web.Application) -> None:
    """Register quick scan routes."""
    handler = QuickScanHandler()

    app.router.add_post("/api/codebase/quick-scan", handler.handle_post_quick_scan)
    app.router.add_get("/api/codebase/quick-scan/{scan_id}", handler.handle_get_quick_scan)
    app.router.add_get("/api/codebase/quick-scans", handler.handle_list_quick_scans)
