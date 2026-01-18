"""
Enterprise Connectors API Handler.

Provides management and monitoring of enterprise data source connectors:
- List available and configured connectors
- Configure connector credentials
- Start/stop sync operations
- View sync history and statistics

Usage:
    GET    /api/connectors                    - List all connectors
    GET    /api/connectors/{id}               - Get connector details
    POST   /api/connectors                    - Configure new connector
    PUT    /api/connectors/{id}               - Update connector config
    DELETE /api/connectors/{id}               - Remove connector
    POST   /api/connectors/{id}/sync          - Start sync
    POST   /api/connectors/sync/{sync_id}/cancel - Cancel running sync
    POST   /api/connectors/test               - Test connection
    GET    /api/connectors/sync-history       - Get sync history
    GET    /api/connectors/stats              - Get aggregate stats
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from aragora.server.handlers.base import BaseHandler

logger = logging.getLogger(__name__)


# In-memory storage (would use database in production)
_connectors: Dict[str, Dict[str, Any]] = {}
_sync_jobs: Dict[str, Dict[str, Any]] = {}
_sync_history: List[Dict[str, Any]] = []


# Connector type metadata
CONNECTOR_TYPES = {
    "github": {
        "name": "GitHub Enterprise",
        "description": "Sync repositories, issues, and pull requests from GitHub",
        "category": "git",
    },
    "s3": {
        "name": "Amazon S3",
        "description": "Index documents from S3 buckets",
        "category": "documents",
    },
    "sharepoint": {
        "name": "Microsoft SharePoint",
        "description": "Sync document libraries from SharePoint Online",
        "category": "documents",
    },
    "postgresql": {
        "name": "PostgreSQL",
        "description": "Sync data from PostgreSQL databases",
        "category": "database",
    },
    "mongodb": {
        "name": "MongoDB",
        "description": "Index collections from MongoDB",
        "category": "database",
    },
    "confluence": {
        "name": "Atlassian Confluence",
        "description": "Index spaces and pages from Confluence",
        "category": "collaboration",
    },
    "notion": {
        "name": "Notion",
        "description": "Sync workspaces and databases from Notion",
        "category": "collaboration",
    },
    "slack": {
        "name": "Slack",
        "description": "Index channel messages and threads",
        "category": "collaboration",
    },
    "fhir": {
        "name": "FHIR (Healthcare)",
        "description": "Connect to FHIR-compliant healthcare systems",
        "category": "healthcare",
    },
    "gdrive": {
        "name": "Google Drive",
        "description": "Sync documents from Google Drive",
        "category": "documents",
        "coming_soon": True,
    },
}


class ConnectorsHandler(BaseHandler):
    """
    Handler for enterprise connector endpoints.

    Provides CRUD operations and sync management for data source connectors.
    """

    ROUTES = [
        "/api/connectors",
        "/api/connectors/{connector_id}",
        "/api/connectors/{connector_id}/sync",
        "/api/connectors/sync/{sync_id}/cancel",
        "/api/connectors/test",
        "/api/connectors/sync-history",
        "/api/connectors/stats",
        "/api/connectors/types",
    ]

    async def handle_request(self, request: Any) -> Any:
        """Route request to appropriate handler."""
        method = request.method
        path = str(request.path)

        # Parse IDs from path
        connector_id = None
        sync_id = None

        if "/connectors/" in path and "/sync" not in path:
            parts = path.split("/connectors/")
            if len(parts) > 1:
                remaining = parts[1].split("/")
                connector_id = remaining[0]
        elif "/sync/" in path:
            parts = path.split("/sync/")
            if len(parts) > 1:
                remaining = parts[1].split("/")
                sync_id = remaining[0]

        # Route to appropriate handler
        if path.endswith("/connectors") and method == "GET":
            return await self._list_connectors(request)
        elif path.endswith("/connectors") and method == "POST":
            return await self._create_connector(request)
        elif path.endswith("/types"):
            return await self._list_types(request)
        elif path.endswith("/sync-history"):
            return await self._get_sync_history(request)
        elif path.endswith("/stats"):
            return await self._get_stats(request)
        elif path.endswith("/test") and method == "POST":
            return await self._test_connection(request)
        elif sync_id and path.endswith("/cancel"):
            return await self._cancel_sync(request, sync_id)
        elif connector_id and path.endswith("/sync") and method == "POST":
            return await self._start_sync(request, connector_id)
        elif connector_id and method == "GET":
            return await self._get_connector(request, connector_id)
        elif connector_id and method == "PUT":
            return await self._update_connector(request, connector_id)
        elif connector_id and method == "DELETE":
            return await self._delete_connector(request, connector_id)

        return self._error_response(404, "Endpoint not found")

    async def _list_connectors(self, request: Any) -> Dict[str, Any]:
        """
        List all configured connectors.

        Query params:
        - status: filter by status (connected, disconnected, syncing, error)
        - type: filter by connector type
        - category: filter by category (git, documents, database, collaboration)
        """
        status_filter = request.query.get("status")
        type_filter = request.query.get("type")
        category_filter = request.query.get("category")

        connectors = list(_connectors.values())

        # Add active sync info
        for connector in connectors:
            active_sync = next(
                (s for s in _sync_jobs.values() if s["connector_id"] == connector["id"] and s["status"] == "running"),
                None,
            )
            if active_sync:
                connector["status"] = "syncing"
                connector["sync_progress"] = active_sync.get("progress", 0)

        # Apply filters
        if status_filter:
            connectors = [c for c in connectors if c["status"] == status_filter]
        if type_filter:
            connectors = [c for c in connectors if c["type"] == type_filter]
        if category_filter:
            connectors = [
                c for c in connectors if CONNECTOR_TYPES.get(c["type"], {}).get("category") == category_filter
            ]

        return self._json_response(
            200,
            {
                "connectors": connectors,
                "total": len(connectors),
                "connected": sum(1 for c in connectors if c["status"] in ("connected", "syncing")),
                "disconnected": sum(1 for c in connectors if c["status"] == "disconnected"),
                "errors": sum(1 for c in connectors if c["status"] == "error"),
            },
        )

    async def _get_connector(self, request: Any, connector_id: str) -> Dict[str, Any]:
        """Get details for a specific connector."""
        connector = _connectors.get(connector_id)
        if not connector:
            return self._error_response(404, f"Connector {connector_id} not found")

        # Add type metadata
        type_meta = CONNECTOR_TYPES.get(connector["type"], {})
        connector["type_name"] = type_meta.get("name", connector["type"])
        connector["category"] = type_meta.get("category", "unknown")

        # Add recent sync history
        connector["recent_syncs"] = [
            s for s in _sync_history if s["connector_id"] == connector_id
        ][-5:]

        return self._json_response(200, connector)

    async def _create_connector(self, request: Any) -> Dict[str, Any]:
        """Configure a new connector."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        connector_type = body.get("type")
        if not connector_type:
            return self._error_response(400, "Connector type is required")

        if connector_type not in CONNECTOR_TYPES:
            return self._error_response(400, f"Unknown connector type: {connector_type}")

        if CONNECTOR_TYPES[connector_type].get("coming_soon"):
            return self._error_response(400, f"Connector type {connector_type} is coming soon")

        # Create connector
        connector_id = str(uuid4())
        type_meta = CONNECTOR_TYPES[connector_type]

        connector = {
            "id": connector_id,
            "type": connector_type,
            "name": body.get("name", type_meta["name"]),
            "description": type_meta["description"],
            "status": "disconnected",
            "config": body.get("config", {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "items_synced": 0,
            "last_sync": None,
        }

        _connectors[connector_id] = connector

        logger.info(f"Created connector {connector_id} of type {connector_type}")

        return self._json_response(201, connector)

    async def _update_connector(self, request: Any, connector_id: str) -> Dict[str, Any]:
        """Update connector configuration."""
        connector = _connectors.get(connector_id)
        if not connector:
            return self._error_response(404, f"Connector {connector_id} not found")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        # Update allowed fields
        if "name" in body:
            connector["name"] = body["name"]
        if "config" in body:
            connector["config"].update(body["config"])

        connector["updated_at"] = datetime.now(timezone.utc).isoformat()

        # If config is updated and was previously connected, mark as needing reconnection
        if "config" in body and connector["status"] == "connected":
            connector["status"] = "configuring"

        _connectors[connector_id] = connector

        logger.info(f"Updated connector {connector_id}")

        return self._json_response(200, connector)

    async def _delete_connector(self, request: Any, connector_id: str) -> Dict[str, Any]:
        """Remove a connector (doesn't delete synced data)."""
        if connector_id not in _connectors:
            return self._error_response(404, f"Connector {connector_id} not found")

        # Cancel any active syncs
        for sync_id, sync_job in list(_sync_jobs.items()):
            if sync_job["connector_id"] == connector_id and sync_job["status"] == "running":
                sync_job["status"] = "cancelled"
                sync_job["completed_at"] = datetime.now(timezone.utc).isoformat()

        del _connectors[connector_id]

        logger.info(f"Deleted connector {connector_id}")

        return self._json_response(200, {"message": "Connector removed successfully"})

    async def _start_sync(self, request: Any, connector_id: str) -> Dict[str, Any]:
        """Start a sync operation for a connector."""
        connector = _connectors.get(connector_id)
        if not connector:
            return self._error_response(404, f"Connector {connector_id} not found")

        # Check if already syncing
        active_sync = next(
            (s for s in _sync_jobs.values() if s["connector_id"] == connector_id and s["status"] == "running"),
            None,
        )
        if active_sync:
            return self._error_response(409, "Sync already in progress")

        # Create sync job
        sync_id = str(uuid4())
        sync_job = {
            "id": sync_id,
            "connector_id": connector_id,
            "connector_name": connector["name"],
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "items_processed": 0,
            "items_total": None,
            "progress": 0,
            "error_message": None,
        }

        _sync_jobs[sync_id] = sync_job
        connector["status"] = "syncing"

        # Start background sync task
        asyncio.create_task(self._run_sync(sync_id, connector_id))

        logger.info(f"Started sync {sync_id} for connector {connector_id}")

        return self._json_response(
            202,
            {
                "message": "Sync started",
                "sync_id": sync_id,
                "connector_id": connector_id,
            },
        )

    async def _run_sync(self, sync_id: str, connector_id: str) -> None:
        """Background task to run sync operation."""
        sync_job = _sync_jobs.get(sync_id)
        connector = _connectors.get(connector_id)

        if not sync_job or not connector:
            return

        try:
            # Simulate sync progress
            total_items = 100 + int(uuid4().int % 1000)
            sync_job["items_total"] = total_items

            for i in range(total_items):
                # Check if cancelled
                if sync_job["status"] == "cancelled":
                    break

                # Simulate processing
                await asyncio.sleep(0.05)
                sync_job["items_processed"] = i + 1
                sync_job["progress"] = (i + 1) / total_items

            # Mark as complete
            if sync_job["status"] != "cancelled":
                sync_job["status"] = "completed"
                connector["status"] = "connected"
                connector["last_sync"] = datetime.now(timezone.utc).isoformat()
                connector["items_synced"] = connector.get("items_synced", 0) + sync_job["items_processed"]

            sync_job["completed_at"] = datetime.now(timezone.utc).isoformat()
            duration = (
                datetime.fromisoformat(sync_job["completed_at"].replace("Z", "+00:00"))
                - datetime.fromisoformat(sync_job["started_at"].replace("Z", "+00:00"))
            ).total_seconds()
            sync_job["duration_seconds"] = int(duration)

            # Add to history
            _sync_history.append(dict(sync_job))

            logger.info(f"Completed sync {sync_id} for connector {connector_id}")

        except Exception as e:
            sync_job["status"] = "failed"
            sync_job["error_message"] = str(e)
            sync_job["completed_at"] = datetime.now(timezone.utc).isoformat()
            connector["status"] = "error"
            connector["error_message"] = str(e)
            _sync_history.append(dict(sync_job))

            logger.error(f"Sync {sync_id} failed: {e}")

    async def _cancel_sync(self, request: Any, sync_id: str) -> Dict[str, Any]:
        """Cancel a running sync operation."""
        sync_job = _sync_jobs.get(sync_id)
        if not sync_job:
            return self._error_response(404, f"Sync job {sync_id} not found")

        if sync_job["status"] != "running":
            return self._error_response(400, "Sync is not running")

        sync_job["status"] = "cancelled"

        # Update connector status
        connector = _connectors.get(sync_job["connector_id"])
        if connector:
            connector["status"] = "connected" if connector.get("last_sync") else "disconnected"

        logger.info(f"Cancelled sync {sync_id}")

        return self._json_response(200, {"message": "Sync cancelled"})

    async def _test_connection(self, request: Any) -> Dict[str, Any]:
        """Test a connector configuration without saving."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        connector_id = body.get("connector_id")
        config = body.get("config", {})

        # In production, this would actually test the connection
        # For now, simulate a test
        await asyncio.sleep(1)

        # Simulate success/failure based on config
        has_required_fields = bool(config)
        success = has_required_fields

        return self._json_response(
            200,
            {
                "success": success,
                "message": "Connection successful" if success else "Missing required configuration",
                "connector_id": connector_id,
            },
        )

    async def _get_sync_history(self, request: Any) -> Dict[str, Any]:
        """Get sync history for all connectors."""
        connector_id = request.query.get("connector_id")
        limit = int(request.query.get("limit", 50))

        history = _sync_history.copy()

        if connector_id:
            history = [h for h in history if h["connector_id"] == connector_id]

        # Sort by start time descending
        history.sort(key=lambda x: x["started_at"], reverse=True)

        # Apply limit
        history = history[:limit]

        return self._json_response(
            200,
            {
                "history": history,
                "total": len(history),
            },
        )

    async def _get_stats(self, request: Any) -> Dict[str, Any]:
        """Get aggregate statistics for all connectors."""
        connectors = list(_connectors.values())

        total_items = sum(c.get("items_synced", 0) for c in connectors)
        connected = sum(1 for c in connectors if c["status"] in ("connected", "syncing"))
        syncing = sum(1 for c in connectors if c["status"] == "syncing")
        errors = sum(1 for c in connectors if c["status"] == "error")

        # Calculate syncs in last 24h
        from datetime import timedelta

        one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
        recent_syncs = [
            h
            for h in _sync_history
            if datetime.fromisoformat(h["started_at"].replace("Z", "+00:00")) >= one_day_ago
        ]

        successful_syncs = sum(1 for s in recent_syncs if s["status"] == "completed")
        failed_syncs = sum(1 for s in recent_syncs if s["status"] == "failed")

        return self._json_response(
            200,
            {
                "total_connectors": len(connectors),
                "connected": connected,
                "syncing": syncing,
                "errors": errors,
                "total_items_synced": total_items,
                "syncs_last_24h": len(recent_syncs),
                "successful_syncs_24h": successful_syncs,
                "failed_syncs_24h": failed_syncs,
                "by_category": self._count_by_category(connectors),
            },
        )

    async def _list_types(self, request: Any) -> Dict[str, Any]:
        """List all available connector types."""
        types = [
            {
                "type": type_id,
                **type_meta,
            }
            for type_id, type_meta in CONNECTOR_TYPES.items()
        ]

        return self._json_response(200, {"types": types})

    def _count_by_category(self, connectors: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count connectors by category."""
        counts: Dict[str, int] = {}
        for connector in connectors:
            category = CONNECTOR_TYPES.get(connector["type"], {}).get("category", "other")
            counts[category] = counts.get(category, 0) + 1
        return counts

    async def _get_json_body(self, request: Any) -> Dict[str, Any]:
        """Parse JSON body from request."""
        body = await request.json()
        return body if isinstance(body, dict) else {}


__all__ = ["ConnectorsHandler"]
