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
from .shared import get_scheduler  # noqa: F401

__all__ = ["ConnectorManagementHandler", "get_scheduler"]
