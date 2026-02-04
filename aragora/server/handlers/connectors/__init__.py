"""Connector Management API Handlers.

Stability: STABLE

Provides unified REST endpoints for discovering, inspecting, and health-checking
all registered connectors via the runtime registry.

Routes:
    GET  /api/v1/connectors              - List all connectors
    GET  /api/v1/connectors/summary      - Aggregated health summary
    GET  /api/v1/connectors/<name>       - Connector detail
    GET  /api/v1/connectors/<name>/health - Health-check a connector
    POST /api/v1/connectors/<name>/test  - Test connector connectivity

Phase Y: Connector Consolidation.
"""

from .management import ConnectorManagementHandler  # noqa: F401
from .shared import _scheduler, get_scheduler  # noqa: F401

# Re-export legacy handler functions for backward compatibility
from .legacy import (  # noqa: F401
    RBAC_AVAILABLE,
    check_permission,  # re-export for test patching
    _check_permission,
    _resolve_tenant_id,
    handle_connector_health,
    handle_create_connector,
    handle_delete_connector,
    handle_get_connector,
    handle_get_scheduler_stats,
    handle_get_sync_history,
    handle_get_sync_status,
    handle_get_workflow_template,
    handle_list_connectors,
    handle_list_workflow_templates,
    handle_mongodb_aggregate,
    handle_mongodb_collections,
    handle_start_scheduler,
    handle_stop_scheduler,
    handle_trigger_sync,
    handle_update_connector,
    handle_webhook,
)

__all__ = [
    "ConnectorManagementHandler",
    "_scheduler",
    "get_scheduler",
    # Legacy exports
    "RBAC_AVAILABLE",
    "check_permission",
    "_check_permission",
    "_resolve_tenant_id",
    "handle_connector_health",
    "handle_create_connector",
    "handle_delete_connector",
    "handle_get_connector",
    "handle_get_scheduler_stats",
    "handle_get_sync_history",
    "handle_get_sync_status",
    "handle_get_workflow_template",
    "handle_list_connectors",
    "handle_list_workflow_templates",
    "handle_mongodb_aggregate",
    "handle_mongodb_collections",
    "handle_start_scheduler",
    "handle_stop_scheduler",
    "handle_trigger_sync",
    "handle_update_connector",
    "handle_webhook",
]
