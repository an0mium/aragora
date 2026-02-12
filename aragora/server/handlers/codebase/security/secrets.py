"""
HTTP API handlers for secrets scanning.

Provides handlers for:
- Trigger secrets scans (current files and git history)
- Get secrets scan results
- List detected secrets
- View secrets scan history
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from aragora.analysis.codebase import SecretsScanResult, SecretsScanner
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    require_permission,
    success_response,
)

from .events import emit_secrets_events
from .storage import (
    get_or_create_secrets_scans,
    get_running_secrets_scans,
    get_secrets_scan_lock,
)

logger = logging.getLogger(__name__)


@require_permission("secrets:scan")
async def handle_scan_secrets(
    repo_path: str,
    repo_id: str | None = None,
    branch: str | None = None,
    include_history: bool = False,
    history_depth: int = 100,
    workspace_id: str | None = None,
    user_id: str | None = None,
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

        running_secrets_scans = get_running_secrets_scans()
        secrets_scan_lock = get_secrets_scan_lock()

        # Check if scan already running
        if repo_id in running_secrets_scans:
            task = running_secrets_scans[repo_id]
            if not task.done():
                return error_response("Secrets scan already in progress", 409)

        # Create initial scan result
        scan_result = SecretsScanResult(
            scan_id=scan_id,
            repository=repo_id,
            branch=branch,
            status="running",
        )

        repo_scans = get_or_create_secrets_scans(repo_id)
        repo_scans[scan_id] = scan_result

        # Start async scan
        async def run_secrets_scan() -> None:
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
                with secrets_scan_lock:
                    result.scan_id = scan_id
                    repo_scans[scan_id] = result

                logger.info(
                    f"[Security] Completed secrets scan {scan_id} for {repo_id}: "
                    f"{len(result.secrets)} secrets found"
                )

                # Emit security events for findings (triggers debate for critical secrets)
                await emit_secrets_events(result, repo_id, scan_id, workspace_id)

            except (OSError, ValueError, TypeError, RuntimeError) as e:
                logger.exception(f"Secrets scan {scan_id} failed: {e}")
                with secrets_scan_lock:
                    scan_result.status = "failed"
                    scan_result.error = str(e)
                    scan_result.completed_at = datetime.now(timezone.utc)

            finally:
                if repo_id in running_secrets_scans:
                    del running_secrets_scans[repo_id]

        # Create and store task
        task = asyncio.create_task(run_secrets_scan())
        running_secrets_scans[repo_id] = task

        logger.info(f"[Security] Started secrets scan {scan_id} for {repo_id}")

        return success_response(
            {
                "scan_id": scan_id,
                "status": "running",
                "repository": repo_id,
                "include_history": include_history,
            }
        )

    except (OSError, ValueError, TypeError, RuntimeError) as e:
        logger.exception(f"Failed to start secrets scan: {e}")
        return error_response(str(e), 500)


@require_permission("secrets:read")
async def handle_get_secrets_scan_status(
    repo_id: str,
    scan_id: str | None = None,
) -> HandlerResult:
    """
    Get secrets scan status/result.

    GET /api/v1/codebase/{repo}/scan/secrets/latest
    GET /api/v1/codebase/{repo}/scan/secrets/{scan_id}
    """
    try:
        repo_scans = get_or_create_secrets_scans(repo_id)

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

    except (KeyError, ValueError, TypeError) as e:
        logger.exception(f"Failed to get secrets scan status: {e}")
        return error_response(str(e), 500)


@require_permission("secrets:read")
async def handle_get_secrets(
    repo_id: str,
    severity: str | None = None,
    secret_type: str | None = None,
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
        repo_scans = get_or_create_secrets_scans(repo_id)

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

        return success_response(
            {
                "secrets": secrets,
                "total": total,
                "limit": limit,
                "offset": offset,
                "scan_id": latest.scan_id,
            }
        )

    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.exception(f"Failed to get secrets: {e}")
        return error_response(str(e), 500)


@require_permission("secrets:read")
async def handle_list_secrets_scans(
    repo_id: str,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> HandlerResult:
    """
    List secrets scan history for a repository.

    GET /api/v1/codebase/{repo}/scans/secrets
    """
    try:
        repo_scans = get_or_create_secrets_scans(repo_id)

        scans = list(repo_scans.values())

        # Filter by status
        if status:
            scans = [s for s in scans if s.status == status]

        # Sort by start time descending
        scans.sort(key=lambda s: s.started_at, reverse=True)

        # Paginate
        total = len(scans)
        scans = scans[offset : offset + limit]

        return success_response(
            {
                "scans": [
                    {
                        "scan_id": s.scan_id,
                        "status": s.status,
                        "started_at": s.started_at.isoformat(),
                        "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                        "files_scanned": s.files_scanned,
                        "scanned_history": s.scanned_history,
                        "history_depth": s.history_depth,
                        "summary": (
                            {
                                "total_secrets": len(s.secrets),
                                "critical_count": s.critical_count,
                                "high_count": s.high_count,
                                "medium_count": s.medium_count,
                                "low_count": s.low_count,
                            }
                            if s.status == "completed"
                            else None
                        ),
                    }
                    for s in scans
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    except (KeyError, ValueError, TypeError) as e:
        logger.exception(f"Failed to list secrets scans: {e}")
        return error_response(str(e), 500)
