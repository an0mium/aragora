"""
Query Operations Mixin for Knowledge Handler.

Provides natural language query handling for the knowledge base.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from aragora.knowledge import QueryOptions
from aragora.server.http_utils import run_async as _run_async

from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)

if TYPE_CHECKING:
    from aragora.knowledge import DatasetQueryEngine, SimpleQueryEngine

logger = logging.getLogger(__name__)


class QueryHandlerProtocol(Protocol):
    """Protocol for handlers that use QueryOperationsMixin."""

    def _get_query_engine(self) -> "DatasetQueryEngine | SimpleQueryEngine": ...


class QueryOperationsMixin:
    """Mixin providing query operations for KnowledgeHandler."""

    @handle_errors("knowledge query")
    def _handle_query(self: QueryHandlerProtocol, query_params: dict, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/query - Natural language query."""

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        question = data.get("question", "")
        if not question:
            return error_response("Question is required", 400)

        workspace_id = data.get("workspace_id", "default")
        options_data = data.get("options", {})

        options = QueryOptions(
            max_chunks=options_data.get("max_chunks", 10),
            search_alpha=options_data.get("search_alpha", 0.5),
            use_agents=options_data.get("use_agents", False),
            extract_facts=options_data.get("extract_facts", True),
            include_citations=options_data.get("include_citations", True),
        )

        engine = self._get_query_engine()

        try:
            result = _run_async(engine.query(question, workspace_id, options))
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return error_response(f"Query failed: {e}", 500)

        return json_response(result.to_dict())
