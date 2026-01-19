"""
Audit Scheduler endpoint handlers.

Endpoints:
- GET  /api/scheduler/jobs - List scheduled jobs
- POST /api/scheduler/jobs - Create a scheduled job
- GET  /api/scheduler/jobs/{job_id} - Get job details
- DELETE /api/scheduler/jobs/{job_id} - Delete a job
- POST /api/scheduler/jobs/{job_id}/trigger - Manually trigger a job
- POST /api/scheduler/jobs/{job_id}/pause - Pause a job
- POST /api/scheduler/jobs/{job_id}/resume - Resume a job
- GET  /api/scheduler/jobs/{job_id}/history - Get job run history
- POST /api/scheduler/webhooks/{webhook_id} - Receive webhook triggers
- POST /api/scheduler/events/git-push - Handle git push events
- POST /api/scheduler/events/file-upload - Handle file upload events
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aragora.server.http_utils import run_async as _run_async

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_user_auth,
)

logger = logging.getLogger(__name__)


class SchedulerHandler(BaseHandler):
    """Handler for audit scheduler endpoints."""

    BASE_ROUTES = [
        "/api/scheduler/jobs",
        "/api/scheduler/webhooks",
        "/api/scheduler/events/git-push",
        "/api/scheduler/events/file-upload",
        "/api/scheduler/status",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.BASE_ROUTES:
            return True
        # Handle dynamic routes
        if path.startswith("/api/scheduler/jobs/"):
            return True
        if path.startswith("/api/scheduler/webhooks/"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        if path == "/api/scheduler/jobs":
            return self._list_jobs(query_params)

        if path == "/api/scheduler/status":
            return self._get_scheduler_status()

        if path.startswith("/api/scheduler/jobs/"):
            parts = path.split("/")
            if len(parts) == 5:
                # /api/scheduler/jobs/{job_id}
                job_id = parts[4]
                return self._get_job(job_id)
            elif len(parts) == 6 and parts[5] == "history":
                # /api/scheduler/jobs/{job_id}/history
                job_id = parts[4]
                limit = int(query_params.get("limit", ["10"])[0])
                return self._get_job_history(job_id, limit)

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        if path == "/api/scheduler/jobs":
            return self._create_job(handler)

        if path == "/api/scheduler/events/git-push":
            return self._handle_git_push(handler)

        if path == "/api/scheduler/events/file-upload":
            return self._handle_file_upload(handler)

        if path.startswith("/api/scheduler/webhooks/"):
            # /api/scheduler/webhooks/{webhook_id}
            parts = path.split("/")
            if len(parts) == 5:
                webhook_id = parts[4]
                return self._handle_webhook(handler, webhook_id)

        if path.startswith("/api/scheduler/jobs/"):
            parts = path.split("/")
            if len(parts) == 6:
                job_id = parts[4]
                action = parts[5]
                if action == "trigger":
                    return self._trigger_job(job_id)
                elif action == "pause":
                    return self._pause_job(job_id)
                elif action == "resume":
                    return self._resume_job(job_id)

        return None

    def handle_delete(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route DELETE requests to appropriate methods."""
        if path.startswith("/api/scheduler/jobs/"):
            parts = path.split("/")
            if len(parts) == 5:
                job_id = parts[4]
                return self._delete_job(job_id)
        return None

    def _get_scheduler(self):
        """Get the scheduler instance."""
        from aragora.scheduler import get_scheduler

        return get_scheduler()

    @require_user_auth
    @handle_errors("list jobs")
    def _list_jobs(self, query_params: dict, user=None) -> HandlerResult:
        """List all scheduled jobs."""
        scheduler = self._get_scheduler()

        status_filter = query_params.get("status", [None])[0]
        workspace_id = query_params.get("workspace_id", [None])[0]

        # Parse status filter
        status = None
        if status_filter:
            from aragora.scheduler import ScheduleStatus

            try:
                status = ScheduleStatus(status_filter)
            except ValueError:
                return error_response(f"Invalid status: {status_filter}", 400)

        jobs = scheduler.list_jobs(status=status, workspace_id=workspace_id)

        return json_response(
            {
                "jobs": [job.to_dict() for job in jobs],
                "count": len(jobs),
            }
        )

    @require_user_auth
    @handle_errors("get job")
    def _get_job(self, job_id: str, user=None) -> HandlerResult:
        """Get details of a specific job."""
        scheduler = self._get_scheduler()
        job = scheduler.get_job(job_id)

        if not job:
            return error_response(f"Job not found: {job_id}", 404)

        return json_response(job.to_dict())

    @require_user_auth
    @handle_errors("create job")
    def _create_job(self, handler, user=None) -> HandlerResult:
        """
        Create a new scheduled job.

        Request body:
        {
            "name": "Daily Security Scan",
            "description": "Scan all documents for security issues",
            "trigger_type": "cron",  // "cron", "interval", "webhook", "git_push", "file_upload"
            "cron": "0 2 * * *",  // For cron trigger
            "interval_minutes": 60,  // For interval trigger
            "preset": "Code Security",  // Or audit_types
            "audit_types": ["security", "compliance"],
            "workspace_id": "ws_123",
            "document_ids": ["doc1", "doc2"],
            "notify_on_complete": true,
            "notify_on_findings": true,
            "finding_severity_threshold": "medium"
        }
        """
        from aragora.scheduler import ScheduleConfig, TriggerType

        body = self.read_json_body(handler)
        if not body:
            return error_response("Request body required", 400)

        name = body.get("name", "").strip()
        if not name:
            return error_response("'name' field is required", 400)

        # Parse trigger type
        trigger_type_str = body.get("trigger_type", "cron")
        try:
            trigger_type = TriggerType(trigger_type_str)
        except ValueError:
            valid = [t.value for t in TriggerType]
            return error_response(f"Invalid trigger_type. Valid: {valid}", 400)

        # Validate trigger-specific fields
        cron = body.get("cron")
        interval_minutes = body.get("interval_minutes")

        if trigger_type == TriggerType.CRON and not cron:
            return error_response("'cron' field required for cron trigger", 400)
        if trigger_type == TriggerType.INTERVAL and not interval_minutes:
            return error_response("'interval_minutes' required for interval trigger", 400)

        # Build config
        config = ScheduleConfig(
            name=name,
            description=body.get("description", ""),
            trigger_type=trigger_type,
            cron=cron,
            interval_minutes=interval_minutes,
            webhook_secret=body.get("webhook_secret"),
            preset=body.get("preset"),
            audit_types=body.get("audit_types", []),
            custom_config=body.get("custom_config", {}),
            workspace_id=body.get("workspace_id"),
            document_ids=body.get("document_ids", []),
            notify_on_complete=body.get("notify_on_complete", True),
            notify_on_findings=body.get("notify_on_findings", True),
            finding_severity_threshold=body.get("finding_severity_threshold", "medium"),
            max_retries=body.get("max_retries", 3),
            timeout_minutes=body.get("timeout_minutes", 60),
            created_by=user.get("id") if user else None,
            tags=body.get("tags", []),
        )

        scheduler = self._get_scheduler()
        job = scheduler.add_schedule(config)

        logger.info(f"Created scheduled job: {job.job_id} ({name})")

        return json_response(
            {
                "success": True,
                "job": job.to_dict(),
            },
            status=201,
        )

    @require_user_auth
    @handle_errors("delete job")
    def _delete_job(self, job_id: str, user=None) -> HandlerResult:
        """Delete a scheduled job."""
        scheduler = self._get_scheduler()

        if not scheduler.get_job(job_id):
            return error_response(f"Job not found: {job_id}", 404)

        success = scheduler.remove_schedule(job_id)

        if success:
            logger.info(f"Deleted scheduled job: {job_id}")
            return json_response({"success": True, "message": f"Job {job_id} deleted"})
        else:
            return error_response(f"Failed to delete job: {job_id}", 500)

    @require_user_auth
    @handle_errors("trigger job")
    def _trigger_job(self, job_id: str, user=None) -> HandlerResult:
        """Manually trigger a job execution."""
        scheduler = self._get_scheduler()

        job = scheduler.get_job(job_id)
        if not job:
            return error_response(f"Job not found: {job_id}", 404)

        # Run the job
        try:
            run = _run_async(scheduler.trigger_job(job_id))
            if run:
                return json_response(
                    {
                        "success": True,
                        "run": run.to_dict(),
                    }
                )
            else:
                return error_response("Failed to trigger job", 500)
        except Exception as e:
            logger.error(f"Failed to trigger job {job_id}: {e}")
            return error_response(f"Failed to trigger job: {str(e)}", 500)

    @require_user_auth
    @handle_errors("pause job")
    def _pause_job(self, job_id: str, user=None) -> HandlerResult:
        """Pause a scheduled job."""
        scheduler = self._get_scheduler()

        if not scheduler.get_job(job_id):
            return error_response(f"Job not found: {job_id}", 404)

        success = scheduler.pause_schedule(job_id)

        if success:
            return json_response({"success": True, "message": f"Job {job_id} paused"})
        else:
            return error_response(f"Could not pause job: {job_id}", 400)

    @require_user_auth
    @handle_errors("resume job")
    def _resume_job(self, job_id: str, user=None) -> HandlerResult:
        """Resume a paused job."""
        scheduler = self._get_scheduler()

        if not scheduler.get_job(job_id):
            return error_response(f"Job not found: {job_id}", 404)

        success = scheduler.resume_schedule(job_id)

        if success:
            return json_response({"success": True, "message": f"Job {job_id} resumed"})
        else:
            return error_response(f"Could not resume job: {job_id}", 400)

    @require_user_auth
    @handle_errors("job history")
    def _get_job_history(self, job_id: str, limit: int = 10, user=None) -> HandlerResult:
        """Get run history for a job."""
        scheduler = self._get_scheduler()

        if not scheduler.get_job(job_id):
            return error_response(f"Job not found: {job_id}", 404)

        history = scheduler.get_job_history(job_id, limit=limit)

        return json_response(
            {
                "job_id": job_id,
                "runs": [run.to_dict() for run in history],
                "count": len(history),
            }
        )

    def _get_scheduler_status(self) -> HandlerResult:
        """Get scheduler status."""
        scheduler = self._get_scheduler()

        jobs = scheduler.list_jobs()
        active_count = sum(1 for j in jobs if j.status.value == "active")
        running_count = sum(1 for j in jobs if j.status.value == "running")

        return json_response(
            {
                "running": scheduler._running,
                "total_jobs": len(jobs),
                "active_jobs": active_count,
                "running_jobs": running_count,
            }
        )

    @handle_errors("webhook")
    def _handle_webhook(self, handler, webhook_id: str) -> HandlerResult:
        """
        Handle incoming webhook trigger.

        Headers:
        - X-Webhook-Signature: HMAC signature for verification
        """
        body = self.read_json_body(handler)
        if not body:
            return error_response("Request body required", 400)

        signature = handler.headers.get("X-Webhook-Signature")

        scheduler = self._get_scheduler()

        try:
            runs = _run_async(
                scheduler.handle_webhook(
                    webhook_id=webhook_id,
                    payload=body,
                    signature=signature,
                )
            )

            return json_response(
                {
                    "success": True,
                    "triggered_jobs": len(runs),
                    "runs": [run.to_dict() for run in runs],
                }
            )
        except Exception as e:
            logger.error(f"Webhook handling failed: {e}")
            return error_response(f"Webhook handling failed: {str(e)}", 500)

    @handle_errors("git push event")
    def _handle_git_push(self, handler) -> HandlerResult:
        """
        Handle git push event (e.g., from GitHub webhook).

        Request body (GitHub format):
        {
            "repository": {"full_name": "owner/repo"},
            "ref": "refs/heads/main",
            "after": "abc123...",
            "commits": [{"modified": [...], "added": [...]}]
        }
        """
        body = self.read_json_body(handler)
        if not body:
            return error_response("Request body required", 400)

        # Parse GitHub webhook format
        repo = body.get("repository", {}).get("full_name", "")
        ref = body.get("ref", "")
        branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref
        commit_sha = body.get("after", "")

        # Collect changed files
        changed_files = []
        for commit in body.get("commits", []):
            changed_files.extend(commit.get("modified", []))
            changed_files.extend(commit.get("added", []))

        scheduler = self._get_scheduler()

        try:
            runs = _run_async(
                scheduler.handle_git_push(
                    repository=repo,
                    branch=branch,
                    commit_sha=commit_sha,
                    changed_files=list(set(changed_files)),
                )
            )

            return json_response(
                {
                    "success": True,
                    "repository": repo,
                    "branch": branch,
                    "triggered_jobs": len(runs),
                    "runs": [run.to_dict() for run in runs],
                }
            )
        except Exception as e:
            logger.error(f"Git push handling failed: {e}")
            return error_response(f"Git push handling failed: {str(e)}", 500)

    @require_user_auth
    @handle_errors("file upload event")
    def _handle_file_upload(self, handler, user=None) -> HandlerResult:
        """
        Handle file upload event to trigger audits.

        Request body:
        {
            "workspace_id": "ws_123",
            "document_ids": ["doc1", "doc2"]
        }
        """
        body = self.read_json_body(handler)
        if not body:
            return error_response("Request body required", 400)

        workspace_id = body.get("workspace_id")
        document_ids = body.get("document_ids", [])

        if not workspace_id:
            return error_response("'workspace_id' required", 400)
        if not document_ids:
            return error_response("'document_ids' required", 400)

        scheduler = self._get_scheduler()

        try:
            runs = _run_async(
                scheduler.handle_file_upload(
                    workspace_id=workspace_id,
                    document_ids=document_ids,
                )
            )

            return json_response(
                {
                    "success": True,
                    "workspace_id": workspace_id,
                    "document_ids": document_ids,
                    "triggered_jobs": len(runs),
                    "runs": [run.to_dict() for run in runs],
                }
            )
        except Exception as e:
            logger.error(f"File upload handling failed: {e}")
            return error_response(f"File upload handling failed: {str(e)}", 500)
