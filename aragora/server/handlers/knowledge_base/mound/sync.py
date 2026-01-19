"""
Sync Operations Mixin for Knowledge Mound Handler.

Provides sync with legacy memory systems:
- Sync from ContinuumMemory
- Sync from ConsensusMemory
- Sync from FactStore
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from aragora.server.http_utils import run_async as _run_async

from ...base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class SyncHandlerProtocol(Protocol):
    """Protocol for handlers that use SyncOperationsMixin."""

    def _get_mound(self) -> "KnowledgeMound | None": ...


class SyncOperationsMixin:
    """Mixin providing sync operations for KnowledgeMoundHandler."""

    @handle_errors("sync continuum")
    def _handle_sync_continuum(self: SyncHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/sync/continuum - Sync from ContinuumMemory."""

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        workspace_id = data.get("workspace_id", "default")
        since = data.get("since")
        limit = data.get("limit", 100)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result = _run_async(
                mound.sync_from_continuum(workspace_id=workspace_id, since=since, limit=limit)
            )
        except AttributeError:
            return json_response({
                "synced": 0,
                "message": "Sync from ContinuumMemory not yet implemented",
                "workspace_id": workspace_id,
            })
        except Exception as e:
            logger.error(f"Failed to sync from continuum: {e}")
            return error_response(f"Failed to sync from continuum: {e}", 500)

        return json_response({
            "synced": result.nodes_synced if hasattr(result, 'nodes_synced') else 0,
            "workspace_id": workspace_id,
            "message": "Sync from ContinuumMemory completed",
        })

    @handle_errors("sync consensus")
    def _handle_sync_consensus(self: SyncHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/sync/consensus - Sync from ConsensusMemory."""

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        workspace_id = data.get("workspace_id", "default")
        since = data.get("since")
        limit = data.get("limit", 100)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result = _run_async(
                mound.sync_from_consensus(workspace_id=workspace_id, since=since, limit=limit)
            )
        except AttributeError:
            return json_response({
                "synced": 0,
                "message": "Sync from ConsensusMemory not yet implemented",
                "workspace_id": workspace_id,
            })
        except Exception as e:
            logger.error(f"Failed to sync from consensus: {e}")
            return error_response(f"Failed to sync from consensus: {e}", 500)

        return json_response({
            "synced": result.nodes_synced if hasattr(result, 'nodes_synced') else 0,
            "workspace_id": workspace_id,
            "message": "Sync from ConsensusMemory completed",
        })

    @handle_errors("sync facts")
    def _handle_sync_facts(self: SyncHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/sync/facts - Sync from FactStore."""

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        workspace_id = data.get("workspace_id", "default")
        since = data.get("since")
        limit = data.get("limit", 100)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result = _run_async(
                mound.sync_from_facts(workspace_id=workspace_id, since=since, limit=limit)
            )
        except AttributeError:
            return json_response({
                "synced": 0,
                "message": "Sync from FactStore not yet implemented",
                "workspace_id": workspace_id,
            })
        except Exception as e:
            logger.error(f"Failed to sync from facts: {e}")
            return error_response(f"Failed to sync from facts: {e}", 500)

        return json_response({
            "synced": result.nodes_synced if hasattr(result, 'nodes_synced') else 0,
            "workspace_id": workspace_id,
            "message": "Sync from FactStore completed",
        })
