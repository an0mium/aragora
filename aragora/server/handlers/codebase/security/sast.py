"""
HTTP API handlers for SAST (Static Application Security Testing).

Provides handlers for:
- Trigger SAST scans with configurable rule sets
- Get SAST scan status and results
- View SAST findings by severity and OWASP category
- Get OWASP Top 10 summary
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    require_permission,
    success_response,
)

from .events import emit_sast_events
from .storage import (
    get_running_sast_scans,
    get_sast_scan_lock,
    get_sast_scan_results,
    get_sast_scanner,
)

logger = logging.getLogger(__name__)


@require_permission("security:sast:scan")
async def handle_scan_sast(
    repo_path: str,
    repo_id: str | None = None,
    rule_sets: list[str] | None = None,
    workspace_id: str | None = None,
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

        running_sast_scans = get_running_sast_scans()
        sast_scan_lock = get_sast_scan_lock()
        sast_scan_results = get_sast_scan_results()

        # Check if scan already running
        if repo_id in running_sast_scans:
            task = running_sast_scans[repo_id]
            if not task.done():
                return error_response("SAST scan already in progress", 409)

        # Get or create storage
        with sast_scan_lock:
            if repo_id not in sast_scan_results:
                sast_scan_results[repo_id] = {}

        # Start async scan
        async def run_sast_scan() -> None:
            try:
                scanner = get_sast_scanner()
                await scanner.initialize()

                result = await scanner.scan_repository(
                    repo_path=repo_path,
                    rule_sets=rule_sets,
                    scan_id=scan_id,
                )

                # Store result
                with sast_scan_lock:
                    sast_scan_results[repo_id][scan_id] = result

                logger.info(
                    "[SAST] Completed scan %s for %s: %s findings",
                    scan_id,
                    repo_id,
                    len(result.findings),
                )

                # Emit security events for critical/high findings
                critical_findings = [
                    f for f in result.findings if f.severity.value in ("critical", "error")
                ]
                if critical_findings:
                    await emit_sast_events(result, repo_id, scan_id, workspace_id)

            except (OSError, ValueError, TypeError, RuntimeError) as e:
                logger.exception("[SAST] Scan failed for %s: %s", repo_id, e)
            finally:
                if repo_id in running_sast_scans:
                    del running_sast_scans[repo_id]

        task = asyncio.create_task(run_sast_scan())
        running_sast_scans[repo_id] = task

        return success_response(
            {
                "message": "SAST scan started",
                "scan_id": scan_id,
                "repo_id": repo_id,
            }
        )

    except (OSError, ValueError, TypeError, RuntimeError) as e:
        logger.exception("[SAST] Failed to start scan: %s", e)
        return error_response("Internal server error", 500)


@require_permission("security:sast:read")
async def handle_get_sast_scan_status(
    repo_id: str,
    scan_id: str,
) -> HandlerResult:
    """Get status and results of a SAST scan."""
    try:
        sast_scan_lock = get_sast_scan_lock()
        sast_scan_results = get_sast_scan_results()
        running_sast_scans = get_running_sast_scans()

        with sast_scan_lock:
            if repo_id not in sast_scan_results:
                return error_response("Repository not found", 404)

            repo_scans = sast_scan_results[repo_id]
            if scan_id not in repo_scans:
                # Check if still running
                if repo_id in running_sast_scans:
                    return success_response(
                        {
                            "scan_id": scan_id,
                            "status": "running",
                            "findings_count": 0,
                        }
                    )
                return error_response("Scan not found", 404)

            result = repo_scans[scan_id]
            return success_response(
                {
                    "scan_id": scan_id,
                    "status": "completed",
                    **result.to_dict(),
                }
            )

    except (KeyError, ValueError, TypeError) as e:
        logger.exception("[SAST] Failed to get scan status: %s", e)
        return error_response("Internal server error", 500)


@require_permission("security:sast:read")
async def handle_get_sast_findings(
    repo_id: str,
    severity: str | None = None,
    owasp_category: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> HandlerResult:
    """Get SAST findings for a repository."""
    try:
        sast_scan_lock = get_sast_scan_lock()
        sast_scan_results = get_sast_scan_results()

        with sast_scan_lock:
            if repo_id not in sast_scan_results:
                return error_response("Repository not found", 404)

            repo_scans = sast_scan_results[repo_id]
            if not repo_scans:
                return success_response(
                    {
                        "findings": [],
                        "total": 0,
                    }
                )

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

            return success_response(
                {
                    "findings": [f.to_dict() for f in findings],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "scan_id": latest_scan.scan_id,
                }
            )

    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.exception("[SAST] Failed to get findings: %s", e)
        return error_response("Internal server error", 500)


@require_permission("security:sast:read")
async def handle_get_owasp_summary(repo_id: str) -> HandlerResult:
    """Get OWASP Top 10 summary for a repository."""
    try:
        sast_scan_lock = get_sast_scan_lock()
        sast_scan_results = get_sast_scan_results()

        with sast_scan_lock:
            if repo_id not in sast_scan_results:
                return error_response("Repository not found", 404)

            repo_scans = sast_scan_results[repo_id]
            if not repo_scans:
                return success_response(
                    {
                        "owasp_summary": {},
                        "total_findings": 0,
                    }
                )

            # Get latest scan
            latest_scan = max(repo_scans.values(), key=lambda s: s.scanned_at)

            # Get OWASP summary
            scanner = get_sast_scanner()
            summary = await scanner.get_owasp_summary(latest_scan.findings)

            return success_response(
                {
                    "scan_id": latest_scan.scan_id,
                    **summary,
                }
            )

    except (KeyError, ValueError, TypeError) as e:
        logger.exception("[SAST] Failed to get OWASP summary: %s", e)
        return error_response("Internal server error", 500)
