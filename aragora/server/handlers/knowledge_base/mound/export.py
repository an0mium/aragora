"""
Export Operations Mixin for Knowledge Mound Handler.

Provides graph export functionality:
- Export as D3 JSON format
- Export as GraphML XML format
- Repository indexing
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from aragora.server.http_utils import run_async as _run_async

from ...base import (
    HandlerResult,
    error_response,
    get_clamped_int_param,
    handle_errors,
    json_response,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class ExportHandlerProtocol(Protocol):
    """Protocol for handlers that use ExportOperationsMixin."""

    def _get_mound(self) -> "KnowledgeMound | None": ...
    def require_auth_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]: ...


class ExportOperationsMixin:
    """Mixin providing export operations for KnowledgeMoundHandler."""

    @handle_errors("D3 graph export")
    def _handle_export_d3(self: ExportHandlerProtocol, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/export/d3 - Export graph as D3 JSON."""
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        start_node_id = query_params.get("start_node_id")
        depth = get_clamped_int_param(query_params, "depth", default=3, min_val=1, max_val=10)
        limit = get_clamped_int_param(query_params, "limit", default=100, min_val=1, max_val=500)

        try:
            result = _run_async(
                mound.export_graph_d3(
                    start_node_id=start_node_id,
                    depth=depth,
                    limit=limit,
                )
            )
        except Exception as e:
            logger.error(f"D3 graph export failed: {e}")
            return error_response(f"D3 graph export failed: {e}", 500)

        return json_response(
            {
                "format": "d3",
                "nodes": result["nodes"],
                "links": result["links"],
                "total_nodes": len(result["nodes"]),
                "total_links": len(result["links"]),
            }
        )

    @handle_errors("GraphML export")
    def _handle_export_graphml(self: ExportHandlerProtocol, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/export/graphml - Export graph as GraphML."""
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        start_node_id = query_params.get("start_node_id")
        depth = get_clamped_int_param(query_params, "depth", default=3, min_val=1, max_val=10)
        limit = get_clamped_int_param(query_params, "limit", default=100, min_val=1, max_val=500)

        try:
            graphml_content = _run_async(
                mound.export_graph_graphml(
                    start_node_id=start_node_id,
                    depth=depth,
                    limit=limit,
                )
            )
        except Exception as e:
            logger.error(f"GraphML export failed: {e}")
            return error_response(f"GraphML export failed: {e}", 500)

        return HandlerResult(
            status_code=200,
            body=graphml_content,
            content_type="application/xml",
        )

    @handle_errors("index repository")
    def _handle_index_repository(self: ExportHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/index/repository - Index a repository."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path is required", 400)

        workspace_id = data.get("workspace_id", "default")

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            from aragora.connectors.repository_crawler import (
                CrawlConfig,
                RepositoryCrawler,
            )

            config = CrawlConfig(
                include_patterns=data.get("include_patterns", ["*", "**/*"]),
                exclude_patterns=data.get(
                    "exclude_patterns",
                    [
                        "**/node_modules/**",
                        "**/.git/**",
                        "**/venv/**",
                        "**/__pycache__/**",
                        "**/.venv/**",
                        "**/dist/**",
                        "**/build/**",
                    ],
                ),
                max_file_size_bytes=data.get("max_file_size_bytes", 1_000_000),
                max_files=data.get("max_files", 10_000),
                extract_symbols=data.get("extract_symbols", True),
                extract_dependencies=data.get("extract_dependencies", True),
                extract_docstrings=data.get("extract_docstrings", True),
            )

            crawler = RepositoryCrawler(config=config, workspace_id=workspace_id)
            crawl_result = _run_async(
                crawler.crawl(repo_path, incremental=data.get("incremental", True))
            )
            nodes_created = _run_async(crawler.index_to_mound(crawl_result, mound))

            return json_response(
                {
                    "status": "completed",
                    "repository": crawl_result.repository_name,
                    "repository_path": crawl_result.repository_path,
                    "workspace_id": workspace_id,
                    "total_files": crawl_result.total_files,
                    "total_lines": crawl_result.total_lines,
                    "total_bytes": crawl_result.total_bytes,
                    "nodes_created": nodes_created,
                    "file_type_counts": crawl_result.file_type_counts,
                    "symbol_counts": crawl_result.symbol_counts,
                    "crawl_duration_ms": crawl_result.crawl_duration_ms,
                    "errors": crawl_result.errors[:10] if crawl_result.errors else [],
                    "warnings": crawl_result.warnings[:10] if crawl_result.warnings else [],
                    "git_info": crawl_result.git_info,
                }
            )

        except FileNotFoundError as e:
            return error_response(f"Repository not found: {e}", 404)
        except Exception as e:
            logger.error(f"Failed to index repository: {e}")
            return error_response(f"Failed to index repository: {e}", 500)
