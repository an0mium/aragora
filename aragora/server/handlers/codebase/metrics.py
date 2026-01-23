"""
HTTP API Handlers for Codebase Metrics Analysis.

Provides REST APIs for code quality metrics:
- Analyze code complexity and maintainability
- Get complexity hotspots
- Detect code duplication
- Track metrics over time

Endpoints:
- POST /api/v1/codebase/{repo}/metrics/analyze - Run metrics analysis
- GET /api/v1/codebase/{repo}/metrics - Get latest metrics
- GET /api/v1/codebase/{repo}/metrics/{analysis_id} - Get specific analysis
- GET /api/v1/codebase/{repo}/hotspots - Get complexity hotspots
- GET /api/v1/codebase/{repo}/duplicates - Get code duplicates
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from aragora.analysis.codebase import (
    CodeMetricsAnalyzer,
    MetricsReport,
)
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Service Registry Integration
# =============================================================================


def _get_metrics_analyzer(
    complexity_warning: int = 10,
    complexity_error: int = 20,
    duplication_threshold: int = 6,
) -> CodeMetricsAnalyzer:
    """Get or create CodeMetricsAnalyzer from service registry."""
    # For analyzers with configurable thresholds, we create new instances
    # but could cache by configuration signature if needed
    return CodeMetricsAnalyzer(
        complexity_warning=complexity_warning,
        complexity_error=complexity_error,
        duplication_threshold=duplication_threshold,
    )


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

_metrics_reports: Dict[str, Dict[str, MetricsReport]] = {}  # repo_id -> {analysis_id -> report}
_metrics_lock = threading.Lock()
_running_analyses: Dict[str, asyncio.Task] = {}


def _get_or_create_repo_metrics(repo_id: str) -> Dict[str, MetricsReport]:
    """Get or create metrics storage for a repository."""
    with _metrics_lock:
        if repo_id not in _metrics_reports:
            _metrics_reports[repo_id] = {}
        return _metrics_reports[repo_id]


# =============================================================================
# Metrics Handlers
# =============================================================================


async def handle_analyze_metrics(
    repo_path: str,
    repo_id: Optional[str] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    complexity_warning: int = 10,
    complexity_error: int = 20,
    duplication_threshold: int = 6,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run metrics analysis on a repository.

    POST /api/v1/codebase/{repo}/metrics/analyze
    {
        "repo_path": "/path/to/repo",
        "include_patterns": ["src/**/*.py"],
        "exclude_patterns": ["**/tests/**"],
        "complexity_warning": 10,
        "complexity_error": 20
    }
    """
    try:
        repo_id = repo_id or f"repo_{uuid.uuid4().hex[:12]}"
        analysis_id = f"metrics_{uuid.uuid4().hex[:12]}"

        # Check if analysis already running
        if repo_id in _running_analyses:
            task = _running_analyses[repo_id]
            if not task.done():
                return {
                    "success": False,
                    "error": "Analysis already in progress",
                    "analysis_id": None,
                }

        # Create placeholder report
        repo_metrics = _get_or_create_repo_metrics(repo_id)

        # Start async analysis
        async def run_analysis():
            try:
                analyzer = _get_metrics_analyzer(
                    complexity_warning=complexity_warning,
                    complexity_error=complexity_error,
                    duplication_threshold=duplication_threshold,
                )

                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                report = await loop.run_in_executor(
                    None,
                    lambda: analyzer.analyze_repository(
                        repo_path=repo_path,
                        scan_id=analysis_id,
                        include_patterns=include_patterns,
                        exclude_patterns=exclude_patterns,
                    ),
                )

                # Store result
                with _metrics_lock:
                    repo_metrics[analysis_id] = report

                logger.info(
                    f"[Metrics] Completed analysis {analysis_id} for {repo_id}: "
                    f"{report.total_files} files, avg complexity {report.avg_complexity:.2f}"
                )

            except Exception as e:
                logger.exception(f"Analysis {analysis_id} failed: {e}")
                # Store error state
                error_report = MetricsReport(
                    repository=repo_id,
                    scan_id=analysis_id,
                )
                with _metrics_lock:
                    repo_metrics[analysis_id] = error_report

            finally:
                if repo_id in _running_analyses:
                    del _running_analyses[repo_id]

        # Create and store task
        task = asyncio.create_task(run_analysis())
        _running_analyses[repo_id] = task

        logger.info(f"[Metrics] Started analysis {analysis_id} for {repo_id}")

        return {
            "success": True,
            "analysis_id": analysis_id,
            "status": "running",
            "repository": repo_id,
        }

    except Exception as e:
        logger.exception(f"Failed to start metrics analysis: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_metrics(
    repo_id: str,
    analysis_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get metrics analysis result.

    GET /api/v1/codebase/{repo}/metrics
    GET /api/v1/codebase/{repo}/metrics/{analysis_id}
    """
    try:
        repo_metrics = _get_or_create_repo_metrics(repo_id)

        if analysis_id:
            # Get specific analysis
            report = repo_metrics.get(analysis_id)
            if not report:
                return {"success": False, "error": "Analysis not found"}
            return {
                "success": True,
                "report": report.to_dict(),
            }
        else:
            # Get latest analysis
            if not repo_metrics:
                return {"success": False, "error": "No analyses found for repository"}

            # Sort by scan time and get latest
            latest = max(repo_metrics.values(), key=lambda r: r.scanned_at)
            return {
                "success": True,
                "report": latest.to_dict(),
            }

    except Exception as e:
        logger.exception(f"Failed to get metrics: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_hotspots(
    repo_id: str,
    min_complexity: int = 5,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Get complexity hotspots from latest analysis.

    GET /api/v1/codebase/{repo}/hotspots
    Query params: min_complexity, limit
    """
    try:
        repo_metrics = _get_or_create_repo_metrics(repo_id)

        if not repo_metrics:
            return {"success": False, "error": "No analyses found for repository"}

        # Get latest analysis
        latest = max(repo_metrics.values(), key=lambda r: r.scanned_at)

        # Filter and sort hotspots
        hotspots = [h for h in latest.hotspots if h.complexity >= min_complexity]
        hotspots.sort(key=lambda h: h.risk_score, reverse=True)
        hotspots = hotspots[:limit]

        return {
            "success": True,
            "hotspots": [h.to_dict() for h in hotspots],
            "total": len(latest.hotspots),
            "analysis_id": latest.scan_id,
        }

    except Exception as e:
        logger.exception(f"Failed to get hotspots: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_duplicates(
    repo_id: str,
    min_lines: int = 6,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Get code duplicates from latest analysis.

    GET /api/v1/codebase/{repo}/duplicates
    Query params: min_lines, limit
    """
    try:
        repo_metrics = _get_or_create_repo_metrics(repo_id)

        if not repo_metrics:
            return {"success": False, "error": "No analyses found for repository"}

        # Get latest analysis
        latest = max(repo_metrics.values(), key=lambda r: r.scanned_at)

        # Filter and sort duplicates
        duplicates = [d for d in latest.duplicates if d.lines >= min_lines]
        duplicates.sort(key=lambda d: d.lines * len(d.occurrences), reverse=True)
        duplicates = duplicates[:limit]

        return {
            "success": True,
            "duplicates": [
                {
                    "hash": d.hash[:8],
                    "lines": d.lines,
                    "occurrences": [
                        {"file": occ[0], "start": occ[1], "end": occ[2]} for occ in d.occurrences
                    ],
                }
                for d in duplicates
            ],
            "total": len(latest.duplicates),
            "analysis_id": latest.scan_id,
        }

    except Exception as e:
        logger.exception(f"Failed to get duplicates: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_file_metrics(
    repo_id: str,
    file_path: str,
) -> Dict[str, Any]:
    """
    Get metrics for a specific file.

    GET /api/v1/codebase/{repo}/metrics/file/{file_path}
    """
    try:
        repo_metrics = _get_or_create_repo_metrics(repo_id)

        if not repo_metrics:
            return {"success": False, "error": "No analyses found for repository"}

        # Get latest analysis
        latest = max(repo_metrics.values(), key=lambda r: r.scanned_at)

        # Find file
        file_metrics = None
        for fm in latest.files:
            if fm.file_path == file_path or fm.file_path.endswith(file_path):
                file_metrics = fm
                break

        if not file_metrics:
            return {"success": False, "error": "File not found in analysis"}

        return {
            "success": True,
            "file": {
                "file_path": file_metrics.file_path,
                "language": file_metrics.language,
                "lines_of_code": file_metrics.lines_of_code,
                "lines_of_comments": file_metrics.lines_of_comments,
                "blank_lines": file_metrics.blank_lines,
                "classes": file_metrics.classes,
                "imports": file_metrics.imports,
                "avg_complexity": file_metrics.avg_complexity,
                "max_complexity": file_metrics.max_complexity,
                "maintainability_index": file_metrics.maintainability_index,
                "functions": [
                    {
                        "name": f.name,
                        "start_line": f.start_line,
                        "end_line": f.end_line,
                        "lines_of_code": f.lines_of_code,
                        "cyclomatic_complexity": f.cyclomatic_complexity,
                        "cognitive_complexity": f.cognitive_complexity,
                        "parameter_count": f.parameter_count,
                        "nested_depth": f.nested_depth,
                    }
                    for f in file_metrics.functions
                ],
            },
            "analysis_id": latest.scan_id,
        }

    except Exception as e:
        logger.exception(f"Failed to get file metrics: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_list_analyses(
    repo_id: str,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    List metrics analysis history for a repository.

    GET /api/v1/codebase/{repo}/metrics/history
    """
    try:
        repo_metrics = _get_or_create_repo_metrics(repo_id)

        analyses = list(repo_metrics.values())

        # Sort by time descending
        analyses.sort(key=lambda r: r.scanned_at, reverse=True)

        # Paginate
        total = len(analyses)
        analyses = analyses[offset : offset + limit]

        return {
            "success": True,
            "analyses": [
                {
                    "analysis_id": r.scan_id,
                    "scanned_at": r.scanned_at.isoformat(),
                    "summary": {
                        "total_files": r.total_files,
                        "total_lines": r.total_lines,
                        "avg_complexity": round(r.avg_complexity, 2),
                        "max_complexity": r.max_complexity,
                        "maintainability_index": round(r.maintainability_index, 2),
                        "hotspot_count": len(r.hotspots),
                        "duplicate_count": len(r.duplicates),
                    },
                }
                for r in analyses
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.exception(f"Failed to list analyses: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Handler Class
# =============================================================================


class MetricsHandler(BaseHandler):
    """
    HTTP handler for codebase metrics endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTE_PREFIXES = [
        "/api/v1/codebase/",
    ]

    def __init__(self, ctx: Dict[str, Any]):
        """Initialize with server context."""
        super().__init__(ctx)  # type: ignore[arg-type]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                # Check for metrics-related paths
                if "/metrics" in path or "/hotspots" in path or "/duplicates" in path:
                    return True
        return False

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route metrics endpoint requests."""
        return None

    async def handle_post_analyze(self, data: Dict[str, Any], repo_id: str) -> HandlerResult:
        """POST /api/v1/codebase/{repo}/metrics/analyze"""
        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path required", 400)

        result = await handle_analyze_metrics(
            repo_path=repo_path,
            repo_id=repo_id,
            include_patterns=data.get("include_patterns"),
            exclude_patterns=data.get("exclude_patterns"),
            complexity_warning=data.get("complexity_warning", 10),
            complexity_error=data.get("complexity_error", 20),
            duplication_threshold=data.get("duplication_threshold", 6),
            workspace_id=data.get("workspace_id"),
            user_id=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_metrics(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/metrics"""
        result = await handle_get_metrics(repo_id=repo_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_get_metrics_by_id(
        self, params: Dict[str, Any], repo_id: str, analysis_id: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/metrics/{analysis_id}"""
        result = await handle_get_metrics(repo_id=repo_id, analysis_id=analysis_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_get_hotspots(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/hotspots"""
        result = await handle_get_hotspots(
            repo_id=repo_id,
            min_complexity=int(params.get("min_complexity", 5)),
            limit=int(params.get("limit", 20)),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_duplicates(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/duplicates"""
        result = await handle_get_duplicates(
            repo_id=repo_id,
            min_lines=int(params.get("min_lines", 6)),
            limit=int(params.get("limit", 20)),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_file_metrics(
        self, params: Dict[str, Any], repo_id: str, file_path: str
    ) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/metrics/file/{file_path}"""
        result = await handle_get_file_metrics(repo_id=repo_id, file_path=file_path)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_list_analyses(self, params: Dict[str, Any], repo_id: str) -> HandlerResult:
        """GET /api/v1/codebase/{repo}/metrics/history"""
        result = await handle_list_analyses(
            repo_id=repo_id,
            limit=int(params.get("limit", 20)),
            offset=int(params.get("offset", 0)),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"
