"""
HTTP API Handlers for Enterprise Connectors.

Provides REST API for managing enterprise data source connectors,
sync operations, and scheduler configuration.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aragora.connectors.enterprise import (
    SyncScheduler,
    SyncSchedule,
    SyncJob,
    SyncHistory,
    GitHubEnterpriseConnector,
    S3Connector,
    PostgreSQLConnector,
    MongoDBConnector,
    FHIRConnector,
)
from aragora.connectors.enterprise.sync.scheduler import SyncStatus

logger = logging.getLogger(__name__)

# Global scheduler instance (initialized on first use)
_scheduler: Optional[SyncScheduler] = None


def get_scheduler() -> SyncScheduler:
    """Get or create the global sync scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SyncScheduler(max_concurrent_syncs=5)
    return _scheduler


# =============================================================================
# Connector Management Handlers
# =============================================================================

async def handle_list_connectors(
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """
    List all registered connectors.

    GET /api/connectors
    """
    scheduler = get_scheduler()
    jobs = scheduler.list_jobs(tenant_id=tenant_id)

    return {
        "connectors": [
            {
                "id": job.connector_id,
                "job_id": job.id,
                "tenant_id": job.tenant_id,
                "schedule": job.schedule.to_dict(),
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "next_run": job.next_run.isoformat() if job.next_run else None,
                "consecutive_failures": job.consecutive_failures,
            }
            for job in jobs
        ],
        "total": len(jobs),
    }


async def handle_get_connector(
    connector_id: str,
    tenant_id: str = "default",
) -> Optional[Dict[str, Any]]:
    """
    Get connector details.

    GET /api/connectors/:connector_id
    """
    scheduler = get_scheduler()
    job_id = f"{tenant_id}:{connector_id}"
    job = scheduler.get_job(job_id)

    if not job:
        return None

    return {
        "id": job.connector_id,
        "job_id": job.id,
        "tenant_id": job.tenant_id,
        "schedule": job.schedule.to_dict(),
        "last_run": job.last_run.isoformat() if job.last_run else None,
        "next_run": job.next_run.isoformat() if job.next_run else None,
        "consecutive_failures": job.consecutive_failures,
        "is_running": job.current_run_id is not None,
    }


async def handle_create_connector(
    connector_type: str,
    config: Dict[str, Any],
    schedule: Optional[Dict[str, Any]] = None,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """
    Create and register a new connector.

    POST /api/connectors
    {
        "type": "github" | "s3" | "postgres" | "mongodb" | "fhir",
        "config": { ... connector-specific config ... },
        "schedule": { ... optional schedule config ... }
    }
    """
    scheduler = get_scheduler()

    # Create connector based on type
    connector = _create_connector(connector_type, config)

    # Create schedule
    sync_schedule = SyncSchedule.from_dict(schedule) if schedule else SyncSchedule()

    # Register with scheduler
    job = scheduler.register_connector(
        connector,
        schedule=sync_schedule,
        tenant_id=tenant_id,
    )

    logger.info(f"Created connector: {connector.connector_id} ({connector_type})")

    return {
        "id": job.connector_id,
        "job_id": job.id,
        "type": connector_type,
        "status": "registered",
    }


async def handle_update_connector(
    connector_id: str,
    updates: Dict[str, Any],
    tenant_id: str = "default",
) -> Optional[Dict[str, Any]]:
    """
    Update connector configuration.

    PATCH /api/connectors/:connector_id
    {
        "schedule": { ... updated schedule ... }
    }
    """
    scheduler = get_scheduler()
    job_id = f"{tenant_id}:{connector_id}"
    job = scheduler.get_job(job_id)

    if not job:
        return None

    # Update schedule if provided
    if "schedule" in updates:
        job.schedule = SyncSchedule.from_dict(updates["schedule"])
        job._calculate_next_run()

    return {
        "id": job.connector_id,
        "status": "updated",
        "schedule": job.schedule.to_dict(),
    }


async def handle_delete_connector(
    connector_id: str,
    tenant_id: str = "default",
) -> bool:
    """
    Delete a connector.

    DELETE /api/connectors/:connector_id
    """
    scheduler = get_scheduler()
    scheduler.unregister_connector(connector_id, tenant_id)
    return True


def _create_connector(connector_type: str, config: Dict[str, Any]):
    """Create a connector instance based on type."""
    if connector_type == "github":
        # GitHubEnterpriseConnector expects repo in "owner/repo" format
        owner = config.get("owner", "")
        repo = config.get("repo", "")
        # If repo already contains owner (owner/repo format), use as-is
        if "/" in repo:
            full_repo = repo
        elif owner and repo:
            full_repo = f"{owner}/{repo}"
        else:
            full_repo = repo or owner  # Fallback

        return GitHubEnterpriseConnector(
            repo=full_repo,
            token=config.get("token"),
            include_prs=config.get("sync_prs", True),
            include_issues=config.get("sync_issues", True),
        )

    elif connector_type == "s3":
        return S3Connector(
            bucket=config["bucket"],
            prefix=config.get("prefix", ""),
            endpoint_url=config.get("endpoint_url"),
            region=config.get("region", "us-east-1"),
        )

    elif connector_type == "postgres":
        return PostgreSQLConnector(
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            database=config["database"],
            schema=config.get("schema", "public"),
            tables=config.get("tables"),
            timestamp_column=config.get("timestamp_column"),
        )

    elif connector_type == "mongodb":
        return MongoDBConnector(
            host=config.get("host", "localhost"),
            port=config.get("port", 27017),
            database=config["database"],
            collections=config.get("collections"),
            connection_string=config.get("connection_string"),
        )

    elif connector_type == "fhir":
        return FHIRConnector(
            base_url=config["base_url"],
            organization_id=config["organization_id"],
            client_id=config.get("client_id"),
            enable_phi_redaction=config.get("enable_phi_redaction", True),
        )

    else:
        raise ValueError(f"Unknown connector type: {connector_type}")


# =============================================================================
# Sync Operation Handlers
# =============================================================================

async def handle_trigger_sync(
    connector_id: str,
    full_sync: bool = False,
    tenant_id: str = "default",
) -> Optional[Dict[str, Any]]:
    """
    Trigger a sync operation.

    POST /api/connectors/:connector_id/sync
    {
        "full_sync": false
    }
    """
    scheduler = get_scheduler()

    run_id = await scheduler.trigger_sync(
        connector_id,
        tenant_id=tenant_id,
        full_sync=full_sync,
    )

    if not run_id:
        return None

    return {
        "run_id": run_id,
        "connector_id": connector_id,
        "status": "started",
        "full_sync": full_sync,
    }


async def handle_get_sync_status(
    connector_id: str,
    tenant_id: str = "default",
) -> Optional[Dict[str, Any]]:
    """
    Get current sync status.

    GET /api/connectors/:connector_id/sync/status
    """
    scheduler = get_scheduler()
    job_id = f"{tenant_id}:{connector_id}"
    job = scheduler.get_job(job_id)

    if not job:
        return None

    return {
        "connector_id": connector_id,
        "is_running": job.current_run_id is not None,
        "current_run_id": job.current_run_id,
        "last_run": job.last_run.isoformat() if job.last_run else None,
        "next_run": job.next_run.isoformat() if job.next_run else None,
        "consecutive_failures": job.consecutive_failures,
    }


async def handle_get_sync_history(
    connector_id: Optional[str] = None,
    tenant_id: str = "default",
    status: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Get sync history.

    GET /api/connectors/sync/history
    GET /api/connectors/:connector_id/sync/history
    """
    scheduler = get_scheduler()

    job_id = f"{tenant_id}:{connector_id}" if connector_id else None
    sync_status = SyncStatus(status) if status else None

    history = scheduler.get_history(
        job_id=job_id,
        tenant_id=tenant_id,
        status=sync_status,
        limit=limit,
    )

    return {
        "history": [h.to_dict() for h in history],
        "total": len(history),
    }


# =============================================================================
# Webhook Handlers
# =============================================================================

async def handle_webhook(
    connector_id: str,
    payload: Dict[str, Any],
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """
    Handle incoming webhook for a connector.

    POST /api/connectors/:connector_id/webhook
    """
    scheduler = get_scheduler()

    handled = await scheduler.handle_webhook(
        connector_id,
        payload,
        tenant_id=tenant_id,
    )

    return {
        "handled": handled,
        "connector_id": connector_id,
    }


# =============================================================================
# Scheduler Handlers
# =============================================================================

async def handle_start_scheduler() -> Dict[str, Any]:
    """
    Start the sync scheduler.

    POST /api/connectors/scheduler/start
    """
    scheduler = get_scheduler()
    await scheduler.start()

    return {
        "status": "started",
    }


async def handle_stop_scheduler() -> Dict[str, Any]:
    """
    Stop the sync scheduler.

    POST /api/connectors/scheduler/stop
    """
    scheduler = get_scheduler()
    await scheduler.stop()

    return {
        "status": "stopped",
    }


async def handle_get_scheduler_stats(
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get scheduler statistics.

    GET /api/connectors/scheduler/stats
    """
    scheduler = get_scheduler()
    return scheduler.get_stats(tenant_id=tenant_id)


# =============================================================================
# Template Handlers
# =============================================================================

async def handle_list_workflow_templates(
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List available workflow templates.

    GET /api/workflows/templates
    GET /api/workflows/templates?category=legal
    """
    from aragora.workflow.templates import list_templates, WORKFLOW_TEMPLATES

    templates = list_templates(category=category)

    return {
        "templates": templates,
        "total": len(templates),
        "categories": list(set(t["category"] for t in templates)),
    }


async def handle_get_workflow_template(
    template_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific workflow template.

    GET /api/workflows/templates/:template_id
    """
    from aragora.workflow.templates import get_template

    template = get_template(template_id)

    if not template:
        return None

    return {
        "id": template_id,
        "template": template,
    }


# =============================================================================
# Health Check
# =============================================================================

async def handle_connector_health() -> Dict[str, Any]:
    """
    Health check for connector subsystem.

    GET /api/connectors/health
    """
    scheduler = get_scheduler()
    stats = scheduler.get_stats()

    return {
        "status": "healthy",
        "scheduler_running": scheduler._scheduler_task is not None,
        "total_connectors": stats["total_jobs"],
        "running_syncs": stats["running_syncs"],
        "success_rate": stats["success_rate"],
    }
